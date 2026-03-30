#!/usr/bin/env python3
"""
Experiment #023: 4h Donchian + 1d EMA Trend + 12h ATR Regime Filter

HYPOTHESIS: Combining 1d EMA for trend direction AND 12h ATR ratio for regime
filter should outperform single-filter approaches:
1. 1d EMA21: overall direction (bull/bear)
2. 12h ATR ratio: confirms trending vs ranging (more responsive than CHOP)
3. 4h Donchian(20): structural breakout entry
4. Volume spike: institutional confirmation

WHY IT SHOULD WORK IN BOTH MARKETS:
- 2021 bull: 1d EMA up + ATR trending → trend-following long entries
- 2022 bear: 1d EMA down + ATR trending → trend-following short entries  
- 2022 crash: 1d EMA down + ATR choppy → no entries (avoids whipsaw)
- 2023-2024 range: 1d EMA flat + ATR choppy → no entries

ENTRY CONDITIONS (4-way confluence):
1. 1d EMA21 rising (bull) or falling (bear) → direction
2. 12h ATR ratio > 1.3 (trending, not ranging) → regime confirm
3. 4h Donchian(20) breakout → precise entry
4. Volume > 1.5x 20-bar MA → institutional confirm

Trade Count Estimate:
- 4h bars/4yr ≈ 8760
- Donchian(20) breakout: ~1 per 40-50 bars = ~175-220 raw
- 1d EMA direction: ~70% qualify = ~122-154
- 12h ATR ratio > 1.3: ~50% qualify = ~61-77
- Volume > 1.5x: ~40% pass = ~24-31 trades/symbol

SAFE RANGE: 24-35 trades over 4 years (on lower end but with higher edge)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_ema_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 12h HTF for regime filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values.astype(np.float64)
    high_12h = df_12h['high'].values.astype(np.float64)
    low_12h = df_12h['low'].values.astype(np.float64)
    
    # ATR ratio on 12h (short-term vs long-term for regime)
    atr_12h_7 = calculate_atr(high_12h, low_12h, close_12h, period=7)
    atr_12h_30 = calculate_atr(high_12h, low_12h, close_12h, period=30)
    atr_ratio_12h = atr_12h_7 / np.where(atr_12h_30 > 1e-10, atr_12h_30, 1.0)
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    
    # === 1d HTF for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values.astype(np.float64)
    ema_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for Donchian20, ATR14, 1d EMA21, 12h ATR ratio
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_ratio_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 12h REGIME FILTER: ATR ratio > 1.3 (trending) ===
        # High ratio = volatility expanding = trending
        atr_regime_trending = atr_ratio_12h_aligned[i] > 1.3
        
        # === 1d TREND DIRECTION: EMA21 slope ===
        ema_curr = ema_1d_aligned[i]
        ema_prev = ema_1d_aligned[i-1] if i > 0 and not np.isnan(ema_1d_aligned[i-1]) else ema_curr
        htf_bull = ema_curr > ema_prev
        htf_bear = ema_curr < ema_prev
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 2 bars ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (entry_price + 2.5 * entry_atr)
            
            reversal_exit = (position_side > 0 and bearish_breakout) or \
                           (position_side < 0 and bullish_breakout)
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold and reversal_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Skip if not trending (range-bound)
            if not atr_regime_trending:
                signals[i] = 0.0
                continue
            
            # LONG: HTF bull + breakout + volume
            if htf_bull and bullish_breakout and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: HTF bear + breakout + volume
            elif htf_bear and bearish_breakout and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals