#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR volatility filter + volume confirmation
# - Long: price breaks above 20-period 4h Donchian high, 1d ATR(14) > 20-period 1d ATR mean, volume > 1.5x 20-period 4h volume SMA
# - Short: price breaks below 20-period 4h Donchian low, same volatility and volume filters
# - Exit: price returns to opposite Donchian level (long exits at Donchian low, short at Donchian high)
# - Uses 1d ATR for volatility regime filter to avoid low-volatility choppy markets
# - Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag
# - Works in both bull and bear markets by capturing breakouts with volatility confirmation

name = "4h_1d_donchian_atr_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for ATR volatility filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d ATR(14) and its 20-period mean for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar TR
    atr_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_mean = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and ATR mean to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1d_mean_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_mean)
    
    # Pre-compute 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_1d_mean_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Donchian levels
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Volatility regime filter: 1d ATR > 20-period 1d ATR mean (avoid low-volatility chop)
        vol_filter = atr_1d_aligned[i] > atr_1d_mean_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above Donchian high with volume and volatility confirmation
        if close_price > donchian_high and vol_confirm and vol_filter:
            enter_long = True
        
        # Short breakout: price closes below Donchian low with volume and volatility confirmation
        if close_price < donchian_low and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions: mean reversion at opposite Donchian level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to or below Donchian low
            exit_long = close_price <= donchian_low
        elif position == -1:
            # Exit short if price rises back to or above Donchian high
            exit_short = close_price >= donchian_high
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals