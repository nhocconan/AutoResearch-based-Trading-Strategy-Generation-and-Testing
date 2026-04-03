#!/usr/bin/env python3
"""
Experiment #154: 1h Donchian(20) breakout + 4h/1d Camarilla pivot + volume confirmation + session filter (08-20 UTC)
HYPOTHESIS: 1h Donchian breakouts aligned with 4h/1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for continuation) 
capture high-probability moves in both bull and bear markets. Volume confirmation (>1.5x average) filters weak breakouts. 
Session filter (08-20 UTC) reduces noise during low-liquidity periods. ATR stoploss (2.0x) manages risk. 
Discrete position sizing (0.20) minimizes fee churn. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_154_1h_donchian20_4h_1d_camarilla_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for Donchian trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    dh_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    dl_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    dh_4h_aligned = align_htf_to_ltf(prices, df_4h, dh_4h)
    dl_4h_aligned = align_htf_to_ltf(prices, df_4h, dl_4h)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    def calculate_camarilla(high, low, close):
        pt = (high + low + close) / 3.0
        rng = high - low
        r3 = pt + rng * 1.1 / 4
        r4 = pt + rng * 1.1 / 2
        s3 = pt - rng * 1.1 / 4
        s4 = pt - rng * 1.1 / 2
        return r3, r4, s3, s4
    
    r3_1d = np.full(len(df_1d), np.nan)
    r4_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 0:
            r3, r4, s3, s4 = calculate_camarilla(
                df_1d['high'].values[i],
                df_1d['low'].values[i],
                df_1d['close'].values[i]
            )
            r3_1d[i] = r3
            r4_1d[i] = r4
            s3_1d[i] = s3
            s4_1d[i] = s4
    
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr_1h = np.zeros(n)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_1h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(dh_4h_aligned[i]) or np.isnan(dl_4h_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- 1h Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- 4h Trend Filter: Only trade in direction of 4h trend ---
        # Long only if price above 4h Donchian middle, short only if below
        trend_middle_4h = (dh_4h_aligned[i] + dl_4h_aligned[i]) / 2.0
        trend_filter_long = price > trend_middle_4h
        trend_filter_short = price < trend_middle_4h
        
        # --- Camarilla Pivot Conditions ---
        near_r3 = abs(price - r3_1d_aligned[i]) / price < 0.005
        near_s3 = abs(price - s3_1d_aligned[i]) / price < 0.005
        break_r4 = price > r4_1d_aligned[i]
        break_s4 = price < s4_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if break_s4 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if break_r4 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: 1h breakout up + volume spike + 4h uptrend + (near R3 or break R4)
        # Short: 1h breakout down + volume spike + 4h downtrend + (near S3 or break S4)
        if (breakout_up and volume_spike and trend_filter_long and (near_r3 or break_r4)):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif (breakout_down and volume_spike and trend_filter_short and (near_s3 or break_s4)):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals