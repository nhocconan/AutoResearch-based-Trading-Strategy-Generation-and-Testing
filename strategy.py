#!/usr/bin/env python3
"""
Experiment #318: 1d Donchian Breakout + 1w Volume Confirmation + Trend Filter

HYPOTHESIS: Daily Donchian channel breakouts (20-period) confirmed by weekly volume spikes 
and aligned with weekly trend (price > weekly EMA50 for longs, < weekly EMA50 for shorts) 
captures medium-term momentum moves. Using weekly data for signal direction and confirmation 
reduces whipsaw and overtrading. Target: 50-100 total trades over 4 years (12-25/year) 
to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (optional for 1d, but keep for consistency)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1w data for volume confirmation and trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 20:
        close_1w = df_1w['close'].values
        volume_1w = df_1w['volume'].values
        
        # Weekly EMA50 for trend filter
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        
        # Weekly volume ratio (current volume / 20-period average)
        vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(volume_1w))
        vol_ratio_1w[20:] = volume_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0
        
        # Align HTF data to LTF (1d)
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 1d Indicators: Donchian Channel (20) ===
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        # Warmup period: use expanding window
        for i in range(20):
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
    
    # === ATR(14) for stoploss (1d) ===
    atr_14 = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate stop loss levels
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian Low (trailing stop)
                if close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian High (trailing stop)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian High with volume confirmation in uptrend
        long_condition = (
            close[i] > donchian_high[i] and 
            vol_ratio_1w_aligned[i] > 1.5 and 
            close[i] > ema_50_1w_aligned[i]
        )
        
        # Short: Break below Donchian Low with volume confirmation in downtrend
        short_condition = (
            close[i] < donchian_low[i] and 
            vol_ratio_1w_aligned[i] > 1.5 and 
            close[i] < ema_50_1w_aligned[i]
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals