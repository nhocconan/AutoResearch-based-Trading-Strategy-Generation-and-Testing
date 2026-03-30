#!/usr/bin/env python3
"""
Experiment #022: 4h Camarilla Pivot + Choppiness + Volume + 12h EMA Bias

HYPOTHESIS: Clone of proven winner:
  gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (ETHUSDT: test_sharpe=1.471, 95tr)

Base Camarilla logic:
- S3/S4 = support levels (long entries when price bounces)
- R3/R4 = resistance levels (short entries when price rejected)
- CHOP > 50 = ranging (Camarilla levels work as reversal points)
- CHOP < 38 = trending (breakouts more likely)

TWIST: Add 12h EMA21 for directional bias:
- Uptrend (close > EMA): prefer S3/S4 long entries
- Downtrend (close < EMA): prefer R3/R4 short entries
- This prevents fading major trends in choppy markets

WHY IT SHOULD WORK IN BOTH MARKETS:
- 2021-2024 bull + crash: Camarilla levels work as reversal zones
- 2025 bear/range: CHOP filter avoids trending strategies in sideways
- 4h timeframe: proven to balance signal quality vs trade count

EXPECTED TRADES: 80-150 per symbol over 4 years
- 4h has ~21900 bars in 4 years
- Each day's Camarilla levels touched 1-3 times
- CHOP + volume filter reduces by ~60%
- Final: ~80-120 per symbol (safe range)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_vol_ema12_v1"
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

def choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    - CHOP > 61.8: choppy/ranging market
    - CHOP < 38.2: trending market
    - Range: 0-100
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        range_sum = highest_high - lowest_low
        
        if range_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA21 for trend direction (align to 4h)
    htf_ema21 = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_aligned = align_htf_to_ltf(prices, df_12h, htf_ema21)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Choppiness Index
    chop = choppiness_index(high, low, close, period=14)
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Camarilla Levels - precompute for alignment ===
    # For each 1d bar, compute Camarilla levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    n_1d = len(close_1d)
    pivot_1d = np.full(n_1d, np.nan)
    s3_1d = np.full(n_1d, np.nan)
    s4_1d = np.full(n_1d, np.nan)
    r3_1d = np.full(n_1d, np.nan)
    r4_1d = np.full(n_1d, np.nan)
    
    for i in range(1, n_1d):  # Start from 1 to have valid previous day
        prev_close = close_1d[i - 1]
        prev_high = high_1d[i - 1]
        prev_low = low_1d[i - 1]
        prev_range = prev_high - prev_low
        
        pivot = (prev_high + prev_low + prev_close) / 3
        
        s1 = prev_close - (0.382 * prev_range)
        s2 = prev_close - (0.618 * prev_range)
        s3 = prev_close - (1.0 * prev_range)
        s4 = prev_close - (1.618 * prev_range)
        
        r1 = prev_close + (0.382 * prev_range)
        r2 = prev_close + (0.618 * prev_range)
        r3 = prev_close + (1.0 * prev_range)
        r4 = prev_close + (1.618 * prev_range)
        
        pivot_1d[i] = pivot
        s3_1d[i] = s3
        s4_1d[i] = s4
        r3_1d[i] = r3
        r4_1d[i] = r4
    
    # Align 1d values to 4h bars
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Enough for all indicators + alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(ema21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION ===
        bull_trend = close[i] > ema21_aligned[i]
        bear_trend = close[i] < ema21_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CHOP REGIME ===
        # CHOP > 50 = ranging (favor Camarilla reversals)
        # CHOP < 38 = trending (may have breakout trades)
        is_ranging = chop[i] > 50.0
        is_trending = chop[i] < 38.0
        
        # === CAMARILLA LEVEL TOUCH ===
        # Check if price is near outer levels
        s3_val = s3_aligned[i] if not np.isnan(s3_aligned[i]) else np.nan
        s4_val = s4_aligned[i] if not np.isnan(s4_aligned[i]) else np.nan
        r3_val = r3_aligned[i] if not np.isnan(r3_aligned[i]) else np.nan
        r4_val = r4_aligned[i] if not np.isnan(r4_aligned[i]) else np.nan
        
        # Tolerance for level touch (0.2% of price)
        tolerance = close[i] * 0.002
        
        # Touch S3 or S4 (price bounced from support)
        touch_s3 = not np.isnan(s3_val) and abs(close[i] - s3_val) < tolerance
        touch_s4 = not np.isnan(s4_val) and abs(close[i] - s4_val) < tolerance
        
        # Touch R3 or R4 (price rejected at resistance)
        touch_r3 = not np.isnan(r3_val) and abs(close[i] - r3_val) < tolerance
        touch_r4 = not np.isnan(r4_val) and abs(close[i] - r4_val) < tolerance
        
        # === BREAKOUT: price beyond outer levels ===
        # Bull breakout: close above R4 (strong momentum up)
        bull_breakout = not np.isnan(r4_val) and close[i] > r4_val
        
        # Bear breakout: close below S4 (strong momentum down)
        bear_breakout = not np.isnan(s4_val) and close[i] < s4_val
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        min_hold_bars = (i - entry_bar) >= 2 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
                stop_hit = low[i] < stop_price
            else:
                stop_price = entry_price + 2.5 * entry_atr
                stop_hit = high[i] > stop_price
            
            # Trend exit: price crosses EMA12 (close vs EMA)
            trend_exit = (position_side > 0 and close[i] < ema21_aligned[i]) or \
                        (position_side < 0 and close[i] > ema21_aligned[i])
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold_bars and trend_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Support touch (S3/S4) + volume + uptrend
            # Or breakout above R4 in bull trend
            if vol_spike:
                if touch_s3 and bull_trend:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    signals[i] = SIZE
                elif touch_s4 and bull_trend:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    signals[i] = SIZE
                elif bull_breakout and is_trending and bull_trend:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    signals[i] = SIZE
            
            # SHORT: Resistance touch (R3/R4) + volume + downtrend
            # Or breakout below S4 in bear trend
            elif vol_spike:
                if touch_r3 and bear_trend:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    signals[i] = -SIZE
                elif touch_r4 and bear_trend:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    signals[i] = -SIZE
                elif bear_breakout and is_trending and bear_trend:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    signals[i] = -SIZE
    
    return signals