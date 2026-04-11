#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: highest high over 20 days
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low over 20 days
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter: 14-period ATR on 1d to filter low volatility
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # ATR percentage of price (to normalize across assets)
    close_1d = df_1d['close'].values
    atr_pct_1d = atr_14_1d / close_1d
    atr_pct_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_pct_1d)
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_pct_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Volatility filter: trade only when ATR% > 0.5% (sufficient volatility)
        vol_filter = atr_pct_1d_aligned[i] > 0.005
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Donchian high + volume confirmation + sufficient volatility
        price_above_high = price_close > donchian_high_1d_aligned[i]
        if price_above_high and vol_confirm and vol_filter:
            enter_long = True
        
        # Short: Price breaks below Donchian low + volume confirmation + sufficient volatility
        price_below_low = price_close < donchian_low_1d_aligned[i]
        if price_below_low and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions: price returns to the midpoint of the Donchian channel
        donchian_mid_1d = (donchian_high_1d + donchian_low_1d) / 2
        donchian_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
        exit_long = price_close < donchian_mid_1d_aligned[i]
        exit_short = price_close > donchian_mid_1d_aligned[i]
        
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

# Hypothesis: Donchian breakout on daily timeframe with volume confirmation and volatility filter.
# Uses 1d Donchian channels (20-day high/low) for breakout signals.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Volatility filter (ATR% > 0.5%) ensures we only trade during sufficient volatility.
# Works in both bull and breakout scenarios by capturing volatility expansion breakouts.
# Reduced position size to 0.25 to manage risk. Target: 20-40 trades/year to minimize fee drag.