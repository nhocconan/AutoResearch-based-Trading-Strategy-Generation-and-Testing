#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (20-day high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1)) if 'close_1d' in locals() else np.abs(high_1d - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1)) if 'close_1d' in locals() else np.abs(low_1d - np.roll(df_1d['close'].values, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: price above/below 1d Donchian mid-point
        donch_mid = (donch_high_20_aligned[i] + donch_low_20_aligned[i]) / 2
        trend_filter_long = price > donch_mid
        trend_filter_short = price < donch_mid
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = vol > (1.5 * vol_avg_20_aligned[i])
        
        # Volatility filter: ATR > 1.5% of price
        vol_filter_atr = atr_14_aligned[i] / price > 0.015 if price > 0 else False
        
        if position == 0:
            # Long setup: price breaks above Donchian high + volume + volatility
            if price > donch_high_20_aligned[i] and vol_filter and vol_filter_atr:
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian low + volume + volatility
            elif price < donch_low_20_aligned[i] and vol_filter and vol_filter_atr:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if price < donch_low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if price > donch_high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dDonchian20_VolVolATR_Breakout_v1"
timeframe = "6h"
leverage = 1.0