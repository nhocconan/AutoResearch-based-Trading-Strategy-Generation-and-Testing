#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian + Williams %R + Choppiness Regime

HYPOTHESIS: Combining Williams %R extremes with Donchian breakout and Choppiness
regime filter creates high-probability entries. Williams %R < -80 = oversold reversal,
combined with Donchian breakout above = momentum shift confirmation.
1w HTF trend keeps us aligned with macro direction.

WHY IT SHOULD WORK:
- Williams %R extremes are proven mean-reversion signals
- Donchian breakout confirms trend acceleration
- Choppiness < 38.2 ensures we only trade in trending markets (avoids 2022 whipsaws)
- 1w HTF SMA keeps us aligned with macro trend

TARGET: 50-80 total over 4 years (12-20/year). Size: 0.30.
Primary: 1d | HTF: 1w
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_williams_chop_1w_v1"
timeframe = "1d"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window_high = np.max(high[i - period + 1:i + 1])
        window_low = np.min(low[i - period + 1:i + 1])
        if window_high != window_low:
            willr[i] = -100 * (window_high - close[i]) / (window_high - window_low)
    
    return willr

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - values < 38.2 = trending, > 61.8 = choppy"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of True Range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        # Highest high - lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        range_sum = highest - lowest
        
        if range_sum > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w SMA for macro direction (call ONCE) ===
    df_1w = get_htf_data(prices, '1w')
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=8, min_periods=8).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Donchian 20 - breakout channel (shift by 1 to avoid look-ahead)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 30  # 1 month for 1d
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND (1w) ===
        htf_bullish = close[i] > sma_1w_aligned[i]
        htf_bearish = close[i] < sma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP < 38.2 = trending (good for breakout trades)
        # CHOP > 61.8 = choppy (avoid - too many false signals)
        is_trending = chop[i] < 38.2 if not np.isnan(chop[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        bullish_breakout = (close[i] > dc_upper_20[i]) if not np.isnan(dc_upper_20[i]) else False
        bearish_breakout = (close[i] < dc_lower_20[i]) if not np.isnan(dc_lower_20[i]) else False
        
        # === WILLIAMS %R MOMENTUM ===
        # Long: %R < -80 (deeply oversold) + breakout = reversal confirmed
        # Short: %R > -20 (overbought) + breakdown = reversal confirmed
        willr_oversold = williams_r[i] < -80 if not np.isnan(williams_r[i]) else False
        willr_overbought = williams_r[i] > -20 if not np.isnan(williams_r[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_ma[i] * 1.5 if vol_ma[i] > 1e-10 else False
        
        # === TRAILING STOP ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 3 bars (3 days) ===
        min_hold = (i - entry_bar) >= 3
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        if in_position:
            stop_hit = False
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Exit on opposite HTF trend (trend reversal)
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: %R oversold + breakout + trending + HTF bullish
            if willr_oversold and bullish_breakout and is_trending and htf_bullish and vol_ok:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: %R overbought + breakdown + trending + HTF bearish
            elif willr_overbought and bearish_breakout and is_trending and htf_bearish and vol_ok:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals