#!/usr/bin/env python3
"""
Experiment #267: 6h Donchian(20) breakout + 1d Weekly Pivot direction + volume confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (price above/below weekly pivot) and confirmed by volume spikes provide high-probability continuation trades. Weekly pivot gives institutional reference point, Donchian captures breakout momentum, volume filters weak moves. Works in bull (breakouts up) and bear (breakouts down) markets by using pivot as dynamic bias. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Weekly Pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Weekly Pivot from 1d data (using prior week's OHLC)
    if len(df_1d) >= 5:
        # Group by week (starting Monday)
        df_1d_df = pd.DataFrame({
            'open': df_1d['open'].values,
            'high': df_1d['high'].values,
            'low': df_1d['low'].values,
            'close': df_1d['close'].values
        }, index=pd.DatetimeIndex(df_1d.index))
        
        # Weekly resample: get prior week's OHLC (shifted by 1 to avoid look-ahead)
        weekly_ohlc = df_1d_df.resample('W-MON').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).shift(1)  # Prior week only
        
        # Calculate Weekly Pivot: P = (H + L + C) / 3
        weekly_high = weekly_ohlc['high'].values
        weekly_low = weekly_ohlc['low'].values
        weekly_close = weekly_ohlc['close'].values
        
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Align to 6h timeframe (forward fill weekly values)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Donchian Channel(20)
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # --- Breakout Detection ---
        breakout_up = close[i] > dc_upper[i]
        breakout_down = close[i] < dc_lower[i]
        
        # --- Weekly Pivot Bias ---
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation ---
        vol_confirm = volume_spike[i]
        
        # --- Entry Logic ---
        # Long: bullish breakout above Donchian upper + price above weekly pivot + volume spike
        if breakout_up and price_above_pivot and vol_confirm:
            signals[i] = SIZE
        # Short: bearish breakout below Donchian lower + price below weekly pivot + volume spike
        elif breakout_down and price_below_pivot and vol_confirm:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals