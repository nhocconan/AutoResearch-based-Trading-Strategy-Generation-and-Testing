#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian Breakout + Choppiness + 1d EMA + Volume

HYPOTHESIS: Breakouts from 4h Donchian channel with volume confirmation
captures institutional moves. 4h Choppiness Index filters out choppy
markets (CHOP > 50 = no trades). 1d EMA aligns with macro trend.
ATR stop at 2.5x controls risk.

WHY BULL + BEAR: 
- Bull 2021: Breakouts in uptrend = strong moves
- Bear 2022: Short breakouts in downtrend = counter-trend but proven edge
- Range 2025: CHOP filter prevents whipsaws in chop

TRADE COUNT: ~75-150 over 4 years (tight = 19-37/year)
Size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_1d_ema_vol_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market choppiness
    CHOP > 61.8 = choppy (no trend)
    CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]), abs(low[idx] - close[idx-1]))
            atr_sum += tr
        
        high_low_range = high[i] - low[i]
        if high_low_range > 1e-10:
            chop[i] = 100 * (np.log(atr_sum) / np.log(high_low_range * period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h indicators for regime filter (call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # 4h ATR for volatility regime
    atr_4h_vals = df_4h['high'] - df_4h['low']
    atr_4h_ma = atr_4h_vals.rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_ma)
    
    # 4h Choppiness Index
    chop_4h_vals = calculate_choppiness(
        df_4h['high'].values, df_4h['low'].values, 
        df_4h['close'].values, period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_vals)
    
    # === LTF: 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio for volatility regime (current vs 4h avg)
    atr_ratio = np.full(n, 1.0)
    valid_mask = (atr_4h_aligned > 1e-10) & ~np.isnan(atr_14)
    atr_ratio[valid_mask] = atr_14[valid_mask] / atr_4h_aligned[valid_mask]
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME FILTERS ===
        # 4h Choppiness: CHOP < 50 = trending (use breakouts)
        chop_trending = not np.isnan(chop_4h_aligned[i]) and chop_4h_aligned[i] < 50.0
        
        # Volatility: ATR ratio > 0.5 (not too compressed)
        vol_ok = atr_ratio[i] > 0.5
        
        # === HTF TREND ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        # === DONCHIAN BREAKOUT CONDITIONS ===
        # Long: price breaks above upper band + HTF uptrend + vol + chop trending
        long_breakout = (close[i] > dc_upper[i]) if not np.isnan(dc_upper[i]) else False
        # Short: price breaks below lower band + HTF downtrend + vol + chop trending
        short_breakout = (close[i] < dc_lower[i]) if not np.isnan(dc_lower[i]) else False
        
        # === MINIMUM HOLD: 2 bars (8h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Opposite HTF trend exits
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Breakout above DC upper + volume + chop trending + HTF bullish
            if long_breakout and vol_spike and chop_trending and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Breakout below DC lower + volume + chop trending + HTF bearish
            elif short_breakout and vol_spike and chop_trending and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals