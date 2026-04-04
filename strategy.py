#!/usr/bin/env python3
"""
Experiment #3334: 1h Volume Spike + 4h Donchian Breakout + 1d EMA Trend Filter
HYPOTHESIS: 1h breakouts triggered by volume spikes (>2.0x 20-period average) capture short-term momentum.
4h Donchian(20) provides medium-term structure/direction, 1d EMA(50) filters for daily trend alignment.
Session filter (08-20 UTC) reduces noise. Position size 0.20. Target: 60-150 total trades over 4 years (15-37/year).
Designed for 1h timeframe: uses HTF for signal direction (4h/1d), 1h only for precise entry timing with volume confirmation.
Works in bull markets (trend continuation) and bear markets (mean reversion from extremes via Donchian channels).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3334_1h_volspike_4h_donchian_1d_ema_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for Donchian channels (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high_4h = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align 4h Donchian to 1h timeframe (auto shift(1) for completed bars only)
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC (pre-compute hours array) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter ---
        hour = hours[i]
        if not (8 <= hour <= 20):  # UTC 08-20
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high_4h_aligned[i]) or np.isnan(lowest_low_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Spike Entry Logic ---
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # 4h Donchian breakout + 1d EMA trend filter
            price_vs_ema = price - ema_1d_aligned[i]
            
            # Long entry: price breaks above 4h Donchian high with bullish daily trend
            if price > highest_high_4h_aligned[i] and price_vs_ema > 0:
                signals[i] = SIZE
            # Short entry: price breaks below 4h Donchian low with bearish daily trend
            elif price < lowest_low_4h_aligned[i] and price_vs_ema < 0:
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals