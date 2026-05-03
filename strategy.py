#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w EMA34 trend filter
# Long when price breaks above 20-day high with volume > 2.0x 5-period average and close > 1w EMA34 (uptrend)
# Short when price breaks below 20-day low with volume > 2.0x 5-period average and close < 1w EMA34 (downtrend)
# Exit on opposite Donchian level or trend failure (close crosses 1w EMA34)
# Uses Donchian for price channel structure, volume for confirmation, 1w EMA34 for trend filter
# Designed for low trade frequency (~7-25/year on 1d) to minimize fee drag
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)

name = "1d_Donchian20_Volume_1wEMA34_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period Donchian high and low (based on previous 20 periods)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (no shift needed as already shifted in calculation)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Load 1w data ONCE before loop for volume spike and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (2.0x 5-period average on 1w)
    vol_ma = pd.Series(df_1w['volume'].values).rolling(window=5, min_periods=5).mean().shift(1).values
    volume_spike = align_htf_to_ltf(prices, df_1w, vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 34, 5) + 1  # Donchian(20) + EMA34(1w) + volume MA(5) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 20-day high with volume spike and close > 1w EMA34 (uptrend)
            if (close[i] > high_20_aligned[i] and 
                volume[i] > volume_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-day low with volume spike and close < 1w EMA34 (downtrend)
            elif (close[i] < low_20_aligned[i] and 
                  volume[i] > volume_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below 20-day low or close < 1w EMA34 (trend failure)
            if (close[i] < low_20_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high or close > 1w EMA34 (trend failure)
            if (close[i] > high_20_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals