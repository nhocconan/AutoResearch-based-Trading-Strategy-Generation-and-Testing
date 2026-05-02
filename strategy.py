#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume spike
# Uses 4h timeframe for signal generation with Williams %R overbought/oversold levels
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
# 1d EMA(34) determines primary trend - only take longs in uptrend, shorts in downtrend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee drag while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Williams %R is effective in ranging markets and catches reversals in bear markets
# Trend filter ensures alignment with higher timeframe direction

name = "4h_WilliamsR_Reversal_1dEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) for trend determination
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R calculation)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) + volume spike + close > 1d EMA34 (bullish trend)
            if williams_r[i] < -80 and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + volume spike + close < 1d EMA34 (bearish trend)
            elif williams_r[i] > -20 and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or close < 1d EMA34 (trend reversal)
            if williams_r[i] > -20 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or close > 1d EMA34 (trend reversal)
            if williams_r[i] < -80 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals