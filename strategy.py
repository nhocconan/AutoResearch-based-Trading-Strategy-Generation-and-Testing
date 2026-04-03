#!/usr/bin/env python3
"""
Experiment #262: 12h Camarilla Pivot + Volume + Chop Regime (1d/1w HTF)
HYPOTHESIS: Camarilla pivot levels (H3/L3) act as strong support/resistance on 1d timeframe. 
On 12h chart, enter long when price breaks above H3 with volume confirmation and chop regime (range), 
enter short when price breaks below L3 with volume confirmation and chop regime. 
Use 1d/1w HTF for regime filter: only trade when 1d ADX < 25 (range) and 1w trend is aligned. 
ATR stoploss (2.5x) manages risk. Discrete position sizing (0.25) minimizes fee drag. 
Target: 75-150 total trades over 4 years (19-37/year). Works in ranging markets via mean reversion at pivots.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_262_12h_camarilla_pivot_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and ADX regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3, H4, L4 from previous 1d bar
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using previous bar's high/low/close for non-look-ahead
    pivot_high = np.roll(high_1d, 1)
    pivot_low = np.roll(low_1d, 1)
    pivot_close = np.roll(close_1d, 1)
    pivot_high[0] = high_1d[0]  # first bar uses current
    pivot_low[0] = low_1d[0]
    pivot_close[0] = close_1d[0]
    
    rang = pivot_high - pivot_low
    h3 = pivot_close + 1.1 * rang / 2
    l3 = pivot_close - 1.1 * rang / 2
    h4 = pivot_close + 1.5 * rang / 2
    l4 = pivot_close - 1.5 * rang / 2
    
    # Align HTF levels to LTF (already shifted by align_htf_to_ltf)
    h3_1d = align_htf_to_ltf(prices, df_1d, h3)
    l3_1d = align_htf_to_ltf(prices, df_1d, l3)
    h4_1d = align_htf_to_ltf(prices, df_1d, h4)
    l4_1d = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1d ADX for regime filter (<25 = range)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === HTF: 1w data for trend alignment ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 1w EMA20 for trend: above = uptrend, below = downtrend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    trend_1w_up = close_1w >= ema_20_1w  # non-aligned for comparison, but we'll align the signal
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(np.float64))
    
    # === 12h Indicators: ATR(14) for stoploss and volume ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(30) for spike detection ===
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[30:] = volume[30:] / vol_ma_30[30:]
    vol_ratio[:30] = 1.0
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high, low, close, period=14):
        atr_sum = np.zeros_like(close)
        true_range = np.zeros_like(close)
        for i in range(len(close)):
            if i == 0:
                true_range[i] = high[i] - low[i]
            else:
                true_range[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if i < period:
                atr_sum[i] = np.sum(true_range[:i+1])
            else:
                atr_sum[i] = np.sum(true_range[i-period+1:i+1])
        max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_regime = (chop > 38.2) & (chop < 61.8)  # range regime
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_1d[i]) or np.isnan(l3_1d[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(trend_1w_up_aligned[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Regime Filters ---
        # 1d ADX < 25 = range (good for mean reversion at pivots)
        adx_range = adx_1d_aligned[i] < 25
        # 1w trend alignment: only trade in direction of weekly trend
        # For longs: weekly uptrend preferred; for shorts: weekly downtrend preferred
        # But we'll allow counter-trend if chop regime is strong
        # Simplified: require chop regime for entry
        in_chop_regime = chop_regime[i]
        
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
                # Exit when price reaches opposite pivot level (mean reversion)
                if price >= h3_1d[i]:
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
                # Exit when price reaches opposite pivot level
                if price <= l3_1d[i]:
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
        if volume_spike and in_chop_regime and adx_range:
            # Long: price breaks above H3 with volume
            if price > h3_1d[i] and close[i-1] <= h3_1d[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below L3 with volume
            elif price < l3_1d[i] and close[i-1] >= l3_1d[i]:
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