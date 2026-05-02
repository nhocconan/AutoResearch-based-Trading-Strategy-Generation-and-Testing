#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets, reversals from extreme levels
# offer high-probability entries. The 1d EMA50 ensures alignment with the higher timeframe trend to avoid
# counter-trend whipsaws. Volume spike (2.0x 20-period average) confirms participation. Discrete sizing 0.25
# targets ~75-150 trades over 4 years (19-38/year) to minimize fee drag. Works in both bull and bear markets
# by fading extremes in the direction of the 1d trend.
# Timeframe: 4h (proven timeframe with good balance of signal quality and trade frequency).

name = "4h_WilliamsR_MeanReversion_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and EMA calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) with 1d uptrend (close > EMA50) and volume spike
            long_entry = williams_r[i] < -80
            # Short entry: Williams %R overbought (> -20) with 1d downtrend (close < EMA50) and volume spike
            short_entry = williams_r[i] > -20
            
            # 1d EMA50 trend filter
            ema_trend_up = close[i] > ema_50_1d_aligned[i]
            ema_trend_down = close[i] < ema_50_1d_aligned[i]
            
            if long_entry and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_entry and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R overbought (> -20) or trend reversal (close < EMA50)
            if williams_r[i] > -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R oversold (< -80) or trend reversal (close > EMA50)
            if williams_r[i] < -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals