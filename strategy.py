#!/usr/bin/env python3
"""
Experiment #114: 1h Donchian(20) breakout + 4h/1d HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: 1h Donchian breakouts in direction of 4h/1d HMA trend with volume confirmation capture medium-term momentum while minimizing overtrading. Uses discrete sizing (0.20), session filter (08-20 UTC), and ATR stoploss (2.0*ATR). Target: 75-150 total trades over 4 years (19-37/year) on 1h timeframe. Works in bull/bear via multi-timeframe trend filter and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_114_1h_donchian20_4h_1d_hma_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for HMA(21) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = pd.Series(df_4h['close'].values)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = close_4h.ewm(span=half_len, adjust=False).mean()
    wma_full = close_4h.ewm(span=21, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_4h = raw_hma.ewm(span=sqrt_len, adjust=False).mean()
    hma_4h_values = hma_4h.values
    daily_trend_4h = np.where(close_4h > hma_4h_values, 1, -1)
    hma_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, daily_trend_4h)
    
    # === HTF: 1d data for HMA(21) trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    wma_half_1d = close_1d.ewm(span=half_len, adjust=False).mean()
    wma_full_1d = close_1d.ewm(span=21, adjust=False).mean()
    raw_hma_1d = 2 * wma_half_1d - wma_full_1d
    hma_1d = raw_hma_1d.ewm(span=sqrt_len, adjust=False).mean()
    hma_1d_values = hma_1d.values
    daily_trend_1d = np.where(close_1d > hma_1d_values, 1, -1)
    hma_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_4h_aligned[i]) or
            np.isnan(hma_trend_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Multi-Timeframe Trend: Require both 4h and 1d bullish/bearish ---
        bullish_trend = (hma_trend_4h_aligned[i] > 0) and (hma_trend_1d_aligned[i] > 0)
        bearish_trend = (hma_trend_4h_aligned[i] < 0) and (hma_trend_1d_aligned[i] < 0)
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~12h on 1h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike and in_session:
            # Long: breakout above upper channel AND bullish 4h/1d trend
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish 4h/1d trend
            elif breakout_down and bearish_trend:
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