#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian(20) breakout + 1w/1d HTF trend filter + volume confirmation
HYPOTHESIS: Price breaking 12h Donchian(20) channels with alignment to 1w EMA50 trend (bullish if price > EMA50, bearish if price < EMA50) and 1d EMA200 filter (avoid counter-trend trades) plus volume confirmation (>1.8x average) captures high-probability institutional breakout flows. The 1w EMA50 provides primary trend bias, while 1d EMA200 acts as a stronger filter to avoid trades against the dominant daily trend. This combination should work in both bull and bear markets by only taking breakouts aligned with the higher timeframe trend. Uses discrete sizing (0.25) and ATR(14) stoploss (2.5) to manage risk. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_028_12h_donchian20_1w_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA50 trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w_50 = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # === HTF: 1d data for EMA200 filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_200 = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # === 12h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    valid_start = 20
    vol_ratio[valid_start:] = volume[valid_start:] / vol_ma[valid_start:]
    vol_ratio[:valid_start] = 1.0
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # sufficient for 200-period EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_1w_50_aligned[i]) or
            np.isnan(ema_1d_200_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Trend Filters ---
        # 1w EMA50: primary trend direction
        trend_1w_bullish = price > ema_1w_50_aligned[i]
        trend_1w_bearish = price < ema_1w_50_aligned[i]
        
        # 1d EMA200: stronger filter to avoid counter-trend trades
        daily_filter_bullish = price > ema_1d_200_aligned[i]
        daily_filter_bearish = price < ema_1d_200_aligned[i]
        
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
            
            # Optional: time-based exit after 4 bars (~48h on 12h) to avoid overtrading
            if bars_since_entry > 4:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND 1w bullish AND 1d bullish filter
            if breakout_up and trend_1w_bullish and daily_filter_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND 1w bearish AND 1d bearish filter
            elif breakout_down and trend_1w_bearish and daily_filter_bearish:
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