#!/usr/bin/env python3
"""
Experiment #074: 1h Camarilla pivot + volume spike + chop regime filter
HYPOTHESIS: 1h Camarilla pivot levels act as intraday support/resistance. 
Long when price breaks above H3 with volume spike in choppy market (CHOP>61.8). 
Short when price breaks below L3 with volume spike in choppy market. 
Uses 4h for trend filter (price > EMA20) and 1d for regime (CHOP>61.8). 
Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
Works in both bull/bear: chop regime filters trending markets, volume confirms breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 20:
        ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    else:
        ema_20_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for chop regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_1d[0] - low_1d[0]  # First period
        atr_1d = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        # Chop = 100 * log10(sum(TR14)/ (max(high14)-min(low14))) / log10(14)
        sum_tr14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        max_h14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_l14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        chop_raw = 100 * (np.log10(sum_tr14) - np.log10(max_h14 - min_l14)) / np.log10(14)
        chop_raw = np.nan_to_num(chop_raw, nan=50.0)
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    else:
        chop_1d_aligned = np.full(n, 50.0)
    
    # === Session filter: 08-20 UTC (pre-compute for efficiency) ===
    hours = prices.index.hour  # Already DatetimeIndex
    
    # === 1h Indicators ===
    # Camarilla pivot levels (based on previous day)
    camarilla_high = np.zeros(n)
    camarilla_low = np.zeros(n)
    # Calculate once per day using previous day's OHLC
    prev_high = np.roll(high, 24)  # 24*1h = 1d ago
    prev_low = np.roll(low, 24)
    prev_close = np.roll(close, 24)
    # Handle first 24 bars
    prev_high[:24] = prev_high[24] if n > 24 else high[0]
    prev_low[:24] = prev_low[24] if n > 24 else low[0]
    prev_close[:24] = prev_close[24] if n > 24 else close[0]
    camarilla_high = prev_close + 1.1 * (prev_high - prev_low) / 6  # H3
    camarilla_low = prev_close - 1.1 * (prev_high - prev_low) / 6   # L3
    
    # Volume confirmation: current vs 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma_20, out=np.ones_like(volume), where=vol_ma_20!=0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in choppy market (CHOP > 61.8) ---
        if chop_1d_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Price above/below 4h EMA20 ---
        price_above_ema = close[i] > ema_20_4h_aligned[i]
        price_below_ema = close[i] < ema_20_4h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Entry Logic ---
        long_condition = (
            close[i] > camarilla_high[i] and 
            price_above_ema and 
            volume_spike
        )
        
        short_condition = (
            close[i] < camarilla_low[i] and 
            price_below_ema and 
            volume_spike
        )
        
        if long_condition:
            signals[i] = SIZE
        elif short_condition:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals