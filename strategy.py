#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1d EMA34 trend filter
# Long when price breaks above 20-period Donchian high with volume > 2.0x 24-bar average and close > 1d EMA34
# Short when price breaks below 20-period Donchian low with volume > 2.0x 24-bar average and close < 1d EMA34
# Exit on opposite Donchian level or trend failure (close crosses 1d EMA34)
# Uses Donchian for price structure, volume for confirmation, 1d EMA34 for trend filter
# Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)

name = "12h_Donchian20_Volume_1dEMA34_v1"
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
    
    # Load 1d data ONCE before loop for Donchian levels and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period Donchian channels on 1d (based on previous day's OHLC)
    # We use the previous completed 1d bar to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(prev_high)
    low_series = pd.Series(prev_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].shift(1).values  # Use previous close to avoid look-ahead
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (2.0x 24-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20, 24) + 1  # EMA34(1d) + Donchian(20) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high with volume spike and close > 1d EMA34 (uptrend)
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume spike and close < 1d EMA34 (downtrend)
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low or close < 1d EMA34 (trend failure)
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or close > 1d EMA34 (trend failure)
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals