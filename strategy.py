#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Daily Donchian channels (20-day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period high and low
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma_20[i]
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above 20-day high with volume
        if price_high > upper_band and volume_confirmed:
            long_signal = True
        
        # Short: price breaks below 20-day low with volume
        if price_low < lower_band and volume_confirmed:
            short_signal = True
        
        # Exit conditions
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 2.5 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 2.5 * atr_val)
        
        # Exit on opposite band touch
        exit_long = position == 1 and price_close < lower_band
        exit_short = position == -1 and price_close > upper_band
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Donchian breakout strategy with volume confirmation and ATR stop loss.
# Enters long when price breaks above 20-day Donchian high with volume confirmation (>1.8x 20-period average volume).
# Enters short when price breaks below 20-day Donchian low with volume confirmation.
# Uses volume confirmation to filter false breakouts and ensure institutional participation.
# Exits when price touches the opposite Donchian band or ATR stop loss (2.5x) is hit.
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets by capturing breakouts in either direction.
# Daily timeframe provides robust support/resistance levels for breakout validation.