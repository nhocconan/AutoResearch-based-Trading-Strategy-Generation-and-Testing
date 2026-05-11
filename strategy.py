#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below Donchian Channel (20) on 12h, filtered by 1d EMA34 trend and volume spike (2x median). Donchian captures breakouts in trending markets, while EMA34 ensures alignment with longer-term momentum. Volume confirms conviction. Designed to work in bull (uptrend breaks) and bear (downtrend breaks). Target: 12-37 trades/year to avoid fee drag.
"""

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 12h Donchian Channel (20) ---
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Filter: spike above 2x median of last 20 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_12h[i] <= entry_price - 2.5 * (donchian_upper[i] - donchian_lower[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.5 * (donchian_upper[i] - donchian_lower[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema34_1d_aligned[i]
        trend_down = close_12h[i] < ema34_1d_aligned[i]
        
        # Volume filter: spike above 2x median
        vol_ok = volume_12h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_12h[i] > donchian_upper[i] and trend_up and vol_ok:
                # Long: price breaks above Donchian upper + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < donchian_lower[i] and trend_down and vol_ok:
                # Short: price breaks below Donchian lower + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss: 2.5x ATR equivalent (using Donchian width as proxy)
                atr_proxy = (donchian_upper[i] - donchian_lower[i]) / 2
                if close_12h[i] <= entry_price - 2.5 * atr_proxy:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below Donchian lower
                elif close_12h[i] <= donchian_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss: 2.5x ATR equivalent
                atr_proxy = (donchian_upper[i] - donchian_lower[i]) / 2
                if close_12h[i] >= entry_price + 2.5 * atr_proxy:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above Donchian upper
                elif close_12h[i] >= donchian_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals