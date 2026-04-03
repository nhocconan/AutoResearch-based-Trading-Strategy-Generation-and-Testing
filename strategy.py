#!/usr/bin/env python3
"""
Experiment #254: 1h Camarilla Pivot + 4h/1d Regime + Volume Spike Strategy

HYPOTHESIS: Camarilla pivot levels on 1h provide precise entry/exit points when aligned with 
higher timeframe trend (4h/1d). In trending regimes (price outside 4h EMA20-EMA50 band), 
we take long at L3 pivot breakout with volume confirmation, short at H3 breakdown. 
In ranging regimes (price between EMAs), we fade extremes at H4/L4 with volume spike. 
Uses ATR-based stoploss (2.0x) and session filter (08-20 UTC) to reduce noise. 
Target: 80-120 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_254_1h_camarilla_pivot_4h_1d_regime_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for regime detection (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA20 and EMA50 for regime filter
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === HTF: 1d data for stronger trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    def calculate_camarilla(high, low, close):
        # Camarilla levels based on previous bar's range
        range_val = high - low
        h5 = close + range_val * 1.1 / 2
        h4 = close + range_val * 1.1 / 4
        h3 = close + range_val * 1.1 / 6
        l3 = close - range_val * 1.1 / 6
        l4 = close - range_val * 1.1 / 4
        l5 = close - range_val * 1.1 / 2
        return h5, h4, h3, l3, l4, l5
    
    # Calculate for previous bar (shifted by 1)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    h5, h4, h3, l3, l4, l5 = calculate_camarilla(high_prev, low_prev, close_prev)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC (pre-compute hours) ===
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # Warmup for 1d EMA200 stability
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filters ---
        # 4h regime: price outside EMA20-EMA50 band = trending
        is_4h_trending = (price > ema20_4h_aligned[i] and price > ema50_4h_aligned[i]) or \
                         (price < ema20_4h_aligned[i] and price < ema50_4h_aligned[i])
        # 1d regime: price outside EMA50-EMA200 band = strong trend
        is_1d_trending = (price > ema50_1d_aligned[i] and price > ema200_1d_aligned[i]) or \
                         (price < ema50_1d_aligned[i] and price < ema200_1d_aligned[i])
        
        # --- Volume Confirmation ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2R (4*ATR profit)
                if high[i] >= entry_price + 4.0 * atr_14[i]:
                    # Scale out 50%
                    signals[i] = position_side * SIZE * 0.5
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2R
                if low[i] <= entry_price - 4.0 * atr_14[i]:
                    signals[i] = position_side * SIZE * 0.5
                    continue
            
            # Exit conditions based on regime
            if position_side > 0:  # Long
                # Exit long if price breaks below L3 (support) OR regime changes to ranging
                if price < l3[i] or not (is_4h_trending and is_1d_trending):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                # Exit short if price breaks above H3 (resistance) OR regime changes to ranging
                if price > h3[i] or not (is_4h_trending and is_1d_trending):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Only enter in strong trending regimes (both 4h and 1d trending)
        if is_4h_trending and is_1d_trending:
            # Long: Price breaks above H3 resistance with volume spike
            if price > h3[i] and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Price breaks below L3 support with volume spike
            elif price < l3[i] and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        # In ranging regimes (price between EMAs), fade extremes at H4/L4
        else:
            # Long fade: Price rejects at L4 support with volume spike
            if price < l4[i] and close[i] > open[i] and volume_spike:  # Bullish rejection
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE * 0.5  # Smaller size for mean reversion
            # Short fade: Price rejects at H4 resistance with volume spike
            elif price > h4[i] and close[i] < open[i] and volume_spike:  # Bearish rejection
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE * 0.5
            else:
                signals[i] = 0.0
    
    return signals