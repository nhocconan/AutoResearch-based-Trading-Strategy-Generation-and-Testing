#!/usr/bin/env python3
"""
Experiment #027: 6h Weekly Pivot + RSI(14) Mean Reversion + Volume

HYPOTHESIS: Combine weekly pivot structure (bull/bear/neutral bias) with 6h RSI
mean reversion signals. This is genuinely different from trend-following approaches:
- Weekly S1 < price < S2 = BULL regime (long bias)
- Weekly R1 > price > R2 = BEAR regime (short bias)
- Weekly between S1 and R1 = NEUTRAL (no trades)
- Within weekly regime: RSI < 30 = long, RSI > 70 = short (mean reversion)
- Volume confirms momentum

This works in BULL: RSI < 30 during pullback → reversal up
This works in BEAR: RSI > 70 during rally → continuation down
This works in RANGE: RSI extremes = mean reversion to center

Target: 60-150 total trades over 4 years. Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_rsi_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_pivot_levels(high, low, close, period='D'):
    """
    Calculate daily pivot points (S1, S2, R1, R2, P)
    Standard formula using previous day's HLC
    """
    n = len(close)
    pivot = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    
    # Use pandas for rolling prev day calculation
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Classic pivot formula
        piv = (prev_high + prev_low + prev_close) / 3.0
        pivot[i] = piv
        
        # Support levels
        s1[i] = 2 * piv - prev_high
        s2[i] = piv - (prev_high - prev_low)
        
        # Resistance levels
        r1[i] = 2 * piv - prev_low
        r2[i] = piv + (prev_high - prev_low)
    
    return pivot, s1, s2, r1, r2

def calculate_rsi(prices, period=14):
    """RSI calculation with min_periods"""
    close = prices.values if hasattr(prices, 'values') else prices
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    high = high.values if hasattr(high, 'values') else high
    low = low.values if hasattr(low, 'values') else low
    close = close.values if hasattr(close, 'values') else close
    
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
    
    # === 6h Indicators ===
    rsi_14 = calculate_rsi(pd.Series(close), period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === HTF: Weekly Pivot (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values.astype(np.float64)
    high_1w = df_1w['high'].values.astype(np.float64)
    low_1w = df_1w['low'].values.astype(np.float64)
    
    # Weekly pivots
    pivot_w, s1_w, s2_w, r1_w, r2_w = calculate_pivot_levels(high_1w, low_1w, close_1w)
    
    # Align weekly to 6h (shift by 1 to avoid look-ahead)
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    
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
    
    warmup = 200  # Need 1 week of HTF data + indicators
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(rsi_14[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(pivot_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or np.isnan(r1_w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY REGIME (direction bias) ===
        # Bull: price above weekly S1
        # Bear: price below weekly R1  
        # Neutral: between S1 and R1 (no trade)
        is_bull = close[i] > s1_w_aligned[i]
        is_bear = close[i] < r1_w_aligned[i]
        is_neutral = not is_bull and not is_bear
        
        # === 6h RSI SIGNALS ===
        # Long: RSI < 30 (oversold)
        # Short: RSI > 70 (overbought)
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
        # === TRAILING STOP UPDATE ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 2 bars (12h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === STOPLOSS CHECK ===
        stop_hit = False
        if in_position:
            if position_side > 0:
                # Long stop: trail from highest
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
                # Exit on regime change to bear
                if min_hold and not is_bull:
                    stop_hit = True
            else:
                # Short stop: trail from lowest
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
                # Exit on regime change to bull
                if min_hold and not is_bear:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # === NEW POSITIONS ===
        # Long: Bull regime + RSI oversold + volume
        if is_bull and rsi_oversold and vol_ok:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        
        # Short: Bear regime + RSI overbought + volume
        elif is_bear and rsi_overbought and vol_ok:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals