#!/usr/bin/env python3
"""
Experiment #1655: 6h Williams %R Reversal + Weekly Volume Spike + Daily Trend Filter
HYPOTHESIS: In 6h timeframe, Williams %R identifies overbought/oversold conditions. Combining with weekly volume spikes (>2x average) and daily EMA50 trend filter captures reversal moves with institutional participation. Weekly volume confirms smart money involvement, daily trend ensures alignment with intermediate-term direction. Target: 75-150 total trades over 4 years (19-37/year) by requiring confluence of three filters. Works in bull/bear markets as reversals occur in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1655_6h_willr_weekly_vol_daily_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1w data for volume spike detection (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values  # ~1 month
    vol_ratio_1w = np.ones(len(volume_1w))
    vol_ratio_1w[4:] = volume_1w[4:] / vol_ma_1w[4:]
    vol_spike_1w = align_htf_to_ltf(prices, df_1w, vol_ratio_1w > 2.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)  # avoid div by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(willr[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(vol_spike_1w[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Williams %R reversal signals
        willr_oversold = willr[i] < -80
        willr_overbought = willr[i] > -20
        
        # Require weekly volume spike and daily trend alignment
        volume_confirmation = vol_spike_1w[i]
        trend_following = (willr_oversold and trend_1d_aligned[i] > 0) or \
                         (willr_overbought and trend_1d_aligned[i] < 0)
        
        if volume_confirmation and trend_following:
            if willr_oversold and trend_1d_aligned[i] > 0:  # Long setup
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif willr_overbought and trend_1d_aligned[i] < 0:  # Short setup
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