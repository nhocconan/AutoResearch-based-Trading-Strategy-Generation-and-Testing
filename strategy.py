#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, close > 1w EMA34, volume > 2.0x 20-bar average
# Short when price breaks below Donchian(20) low, close < 1w EMA34, volume > 2.0x 20-bar average
# Uses Donchian for price channel structure, 1w EMA34 for higher timeframe trend filter, volume for momentum confirmation
# Designed for low trade frequency (~7-25/year on 1d) to minimize fee drag
# Works in bull (breakouts with rising volume in uptrend) and bear (breakdowns with rising volume in downtrend)

name = "1d_Donchian20_Volume_1wEMA34_Trend_v1"
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
    
    # Load 1d data ONCE before loop (for Donchian calculation)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: max(high_1d, window=20)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: min(low_1d, window=20)
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned but keep for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (2.0x 20-period average on 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20) + 1  # EMA34(1w) + Donchian(20) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian high, close > 1w EMA34, volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low, close < 1w EMA34, volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian low or close < 1w EMA34 (trend failure)
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian high or close > 1w EMA34 (trend failure)
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals