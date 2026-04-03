#!/usr/bin/env python3
"""
Experiment #135: 6h Elder Ray + Weekly Trend + Volume Spike
HYPOTHESIS: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13.
In strong trends (ADX>25 on 1w), Elder Ray extremes with volume spikes (>2x) signal continuation.
Weekly trend filter (price > EMA50_1w) ensures alignment with higher timeframe momentum.
This captures sustained moves in both bull and bear markets while avoiding chop.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_135_6h_elder_ray_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 6h Indicators: ADX(14) for trend strength ===
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_dm[i] = high[i] - high[i-1] if high[i] - high[i-1] > high[i-1] - low[i-1] and high[i] - high[i-1] > 0 else 0
        minus_dm[i] = high[i-1] - low[i-1] if high[i-1] - low[i-1] > high[i] - high[i-1] and high[i-1] - low[i-1] > 0 else 0
    
    tr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(adx[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Trend Filter: Weekly Uptrend/Downturn ---
        weekly_uptrend = price > ema50_1w_aligned[i]
        weekly_downtrend = price < ema50_1w_aligned[i]
        
        # --- Elder Ray Extremes with Volume Spike ---
        vol_spike = vol_ratio[i] > 2.0
        strong_bull = bull_power[i] > 2.0 * atr_14[i]  # Strong buying pressure
        strong_bear = bear_power[i] < -2.0 * atr_14[i]  # Strong selling pressure
        
        # --- ADX Trend Strength Filter ---
        trending = adx[i] > 25
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if bull power fades
                if bull_power[i] < 0.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if bear power fades
                if bear_power[i] > -0.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if weekly_uptrend and trending and vol_spike and strong_bull:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif weekly_downtrend and trending and vol_spike and strong_bear:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals