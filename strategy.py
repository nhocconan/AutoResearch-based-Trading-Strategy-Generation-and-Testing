#!/usr/bin/env python3
"""
Experiment #1975: 6h Donchian(20) Breakout + Weekly Trend Filter + Volume Spike
HYPOTHESIS: Weekly trend determines institutional bias (bull/bear), while 6h Donchian breakouts with volume confirmation capture intermediate-term swings. 
Weekly EMA(50) filter avoids counter-trend trades in strong trends. Volume spike (>2x 20-period average) confirms institutional participation. 
Target: 75-150 total trades over 4 years with discrete sizing (0.25) to minimize fee drag.
Works in bull markets via breakouts and bear markets via short breakdowns aligned with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1975_6h_donchian20_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_50_1w, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Donchian(20) channels and Volume MA(20) ===
    # Donchian upper/lower bands (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for weekly EMA(50) and 6h indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Opposite Donchian breakout or time-based exit ---
        if in_position:
            bars_since_entry += 1
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below 6h Donchian lower band (contrarian exit)
                if price < donch_low[i]:
                    exit_signal = True
                # Time-based exit: max 8 bars (2 days) to avoid overtrading
                elif bars_since_entry >= 8:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above 6h Donchian upper band
                if price > donch_high[i]:
                    exit_signal = True
                # Time-based exit: max 8 bars (2 days)
                elif bars_since_entry >= 8:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly trend alignment for bias filter
        weekly_trend = trend_1w_aligned[i]
        
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above 6h Donchian upper band AND weekly uptrend
            if weekly_trend > 0 and price > donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian lower band AND weekly downtrend
            elif weekly_trend < 0 and price < donch_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals