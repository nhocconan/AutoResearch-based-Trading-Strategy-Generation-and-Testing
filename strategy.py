#!/usr/bin/env python3
"""
4h_DonchianBreakout_1dTrend_Volume
Hypothesis: Price breaks beyond Donchian(20) channels on 4h, filtered by 1d EMA34 trend and volume spike. Donchian breakouts capture trend momentum, while 1d EMA filter ensures alignment with longer-term direction. Volume confirmation reduces false breakouts. Designed for 20-40 trades/year per symbol to minimize fee drag while capturing strong trending moves in both bull and bear markets.
"""

name = "4h_DonchianBreakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 4h Donchian Channels (20-period) ---
    # Upper channel: highest high of last 20 periods
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Filter: spike above 1.5x median of last 50 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=50, min_periods=20).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for Donchian and EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss based on ATR-like measure (using Donchian width)
                channel_width = donchian_upper[i] - donchian_lower[i]
                if channel_width > 0:  # avoid division by zero
                    atr_estimate = channel_width / 4  # rough ATR approximation
                    if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_estimate:
                        signals[i] = 0.0
                        position = 0
                    elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_estimate:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema34_1d_aligned[i]
        trend_down = close_4h[i] < ema34_1d_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_4h[i] > donchian_upper[i] and trend_up and vol_ok:
                # Long: price breaks above upper Donchian + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < donchian_lower[i] and trend_down and vol_ok:
                # Short: price breaks below lower Donchian + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Exit conditions: price returns to middle of Donchian channel
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if position == 1:
                # Exit long: price returns to or below middle of channel
                if close_4h[i] <= donchian_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to or above middle of channel
                if close_4h[i] >= donchian_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals