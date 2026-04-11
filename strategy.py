#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_breakout_volume_v1"
timeframe = "6h"
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
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower bands
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 6h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Filter: Avoid choppy markets - require ATR ratio > 0.6
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / (atr_ma_50 + 1e-10)
    trending_market = atr_ratio > 0.6
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(trending_market[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        upper = upper_1d_aligned[i]
        lower = lower_1d_aligned[i]
        atr_val = atr[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry signals - only in trending markets to avoid whipsaws
        long_signal = False
        short_signal = False
        
        # Long: price breaks above daily upper Donchian with volume and trending
        if price_high > upper and volume_confirmed and trending:
            long_signal = True
        
        # Short: price breaks below daily lower Donchian with volume and trending
        if price_low < lower and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Calculate daily midpoint for exit
        midpoint_1d = (upper_1d + lower_1d) / 2
        midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)[i]
        
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr_val)
        
        # Exit to midpoint
        exit_long = position == 1 and price_close < midpoint_aligned
        exit_short = position == -1 and price_close > midpoint_aligned
        
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

# Hypothesis: 6h Donchian breakout strategy with volume confirmation and trend filter.
# Enters long when price breaks above daily 20-period Donchian upper band with volume confirmation (>1.5x avg volume) in trending markets (ATR ratio > 0.6).
# Enters short when price breaks below daily 20-period Donchian lower band with volume confirmation and trending.
# Uses volume confirmation to ensure institutional participation and trend filter to avoid whipsaws in sideways markets.
# Exits when price returns to daily Donchian midpoint or ATR stop loss (2x) is hit.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by trading breakouts in either direction with trend filter.