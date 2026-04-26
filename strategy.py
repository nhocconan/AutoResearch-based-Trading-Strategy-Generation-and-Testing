#!/usr/bin/env python3
"""
6h_Donchian20_VolumeSpike_12hTrend
Hypothesis: Donchian(20) breakouts on 6h with volume confirmation and 12h trend filter capture institutional moves in both bull and bear markets. Long when price breaks above upper band in bullish 12h trend with volume spike; short when breaks below lower band in bearish 12h trend. Uses discrete sizing (±0.25) to limit churn and targets 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA20 for higher-timeframe trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Donchian(20) on 6h: upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of calculations (20 for Donchian, 20 for volume MA, 20 for EMA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        ema_20_val = ema_20_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 12h trend: bullish if price > EMA20, bearish if price < EMA20
        bullish_12h = close_val > ema_20_val
        bearish_12h = close_val < ema_20_val
        
        # Entry conditions: price breaks above/below Donchian bands in direction of 12h trend with volume confirmation
        long_entry = (close_val > upper_val) and bullish_12h and vol_spike
        short_entry = (close_val < lower_val) and bearish_12h and vol_spike
        
        # Exit conditions: price returns inside Donchian bands or trend reversal
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < upper_val or not bullish_12h):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > lower_val or not bearish_12h):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_VolumeSpike_12hTrend"
timeframe = "6h"
leverage = 1.0