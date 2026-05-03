#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold), price > 1d EMA34, and volume > 1.8x 24-bar average
# Short when Williams %R > -20 (overbought), price < 1d EMA34, and volume > 1.8x 24-bar average
# Uses 1d EMA for higher timeframe trend alignment (matches experiment HTF)
# Volume spike confirms reversal strength
# Discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (12-37/year on 12h) to avoid fee drag
# Works in bull (reversals above rising EMA) and bear (reversals below falling EMA)

name = "12h_WilliamsR_Reversal_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1d data for EMA(34) trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d: (highest_high - close) / (highest_high - lowest_low) * -100
    # Williams %R = -100 * (HHV - CLOSE) / (HHV - LLV)
    period = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero (when HHV == LLV)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation (1.8x 24-period average on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 24) + 1  # EMA(34) + Williams %R(14) + volume MA(24) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold), price > 1d EMA34, volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought), price < 1d EMA34, volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or price < 1d EMA34
            if (williams_r_aligned[i] > -20 or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or price > 1d EMA34
            if (williams_r_aligned[i] < -80 or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals