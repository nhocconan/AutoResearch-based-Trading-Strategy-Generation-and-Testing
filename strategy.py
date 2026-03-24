#!/usr/bin/env python3
"""
Experiment #085: 15m Primary + 4h/1d HTF — Camarilla Pivot + RSI + HMA Trend

Hypothesis: After 81 failed experiments, 15m strategies keep generating 0 trades due to
overly strict entry conditions. This strategy uses Camarilla pivot levels which are
TOUCHED MULTIPLE TIMES PER DAY, ensuring trade generation while maintaining selectivity.

Key design choices:
- Timeframe: 15m (target 40-100 trades/year)
- HTF: 4h HMA for trend bias, 1d Camarilla pivot levels for entry zones
- Entry: Price at Camarilla R3/S3 + RSI(7) extreme + 4h trend alignment
- Camarilla levels are hit 2-4x per day on crypto = ensures trade frequency
- RSI(7) is faster than RSI(14) = more entry signals on 15m
- Position size: 0.20 (20% of capital, conservative for 15m frequency)
- Stoploss: 2.5x ATR trailing

Why this should work where others failed:
- Camarilla R3/S3 are MEAN REVERSION levels that price regularly touches
- Unlike Donchian breakouts (rare), Camarilla levels trigger daily
- Loose RSI(7) < 35 or > 65 (not extreme 20/80) = more signals
- 4h HMA bias is soft (just direction, not hard filter) = trades still generate
- Session filter OPTIONAL (disabled by default) to ensure trades on all symbols

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_rsi_hma_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_camarilla_pivots(high, low, close, prev_close):
    """
    Camarilla Pivot Levels
    R4/S4 = breakout levels
    R3/S3 = mean reversion levels (our entry zones)
    R2/S2, R1/S1 = minor levels
    Formula based on previous day's range
    """
    n = len(close)
    
    # Pivot = (H + L + C) / 3
    pivot = (high + low + close) / 3.0
    
    # Range
    range_hl = high - low
    
    # Camarilla levels
    # R4 = C + 1.5 * Range, S4 = C - 1.5 * Range (breakout)
    # R3 = C + 1.0 * Range, S3 = C - 1.0 * Range (mean reversion entry)
    # R2 = C + 0.5 * Range, S2 = C - 0.5 * Range
    # R1 = C + 0.25 * Range, S1 = C - 0.25 * Range
    
    r4 = close + 1.5 * range_hl
    s4 = close - 1.5 * range_hl
    r3 = close + 1.0 * range_hl
    s3 = close - 1.0 * range_hl
    r2 = close + 0.5 * range_hl
    s2 = close - 0.5 * range_hl
    r1 = close + 0.25 * range_hl
    s1 = close - 0.25 * range_hl
    
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d Camarilla pivots and align
    # Use previous day's OHLC for pivot calculation
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_close_1d[0] = df_1d['close'].values[0]
    
    pivot_1d, r1_1d, r2_1d, r3_1d, r4_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        prev_close_1d
    )
    
    # Align all 1d levels to 15m
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # 15m HMA for short-term trend
    hma_15m = calculate_hma(close, period=13)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m frequency)
    SIZE_HALF = 0.10  # Half position for take profit
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === CAMARILLA LEVEL PROXIMITY ===
        # Check if price is near S3 (long zone) or R3 (short zone)
        # Use 1% tolerance for "near"
        tolerance = 0.01  # 1% of price
        
        near_s3 = abs(close[i] - s3_aligned[i]) / (close[i] + 1e-10) < tolerance
        near_r3 = abs(close[i] - r3_aligned[i]) / (close[i] + 1e-10) < tolerance
        
        # Also check if price bounced off S3/R3 (crossed and came back)
        # For simplicity, just check proximity + RSI confirmation
        
        # === RSI(7) EXTREMES (LOOSE for trade generation) ===
        rsi_oversold = rsi_7[i] < 35.0  # Not too extreme, ensures trades
        rsi_overbought = rsi_7[i] > 65.0  # Not too extreme, ensures trades
        
        # === 15m HMA SHORT-TERM TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === DESIRED SIGNAL (Camarilla Mean Reversion + HTF Bias) ===
        desired_signal = 0.0
        
        # LONG: Price at S3 + RSI oversold + 4h not strongly bearish
        # Soft 4h filter: only avoid if strongly against (close << 4h HMA)
        htf_not_strongly_bear = not (htf_bear and (close[i] < hma_4h_aligned[i] * 0.98))
        
        if near_s3 and rsi_oversold and htf_not_strongly_bear:
            desired_signal = SIZE
        # Bonus: if 4h is bullish, increase conviction
        elif near_s3 and rsi_oversold and htf_bull:
            desired_signal = SIZE * 1.2  # Will be capped to SIZE
        
        # SHORT: Price at R3 + RSI overbought + 4h not strongly bullish
        htf_not_strongly_bull = not (htf_bull and (close[i] > hma_4h_aligned[i] * 1.02))
        
        if near_r3 and rsi_overbought and htf_not_strongly_bull:
            desired_signal = -SIZE
        # Bonus: if 4h is bearish, increase conviction
        elif near_r3 and rsi_overbought and htf_bear:
            desired_signal = -SIZE * 1.2  # Will be capped to -SIZE
        
        # === BREAKOUT MODE (if price breaks R4/S4 with momentum) ===
        # Less frequent but higher conviction
        breakout_bull = close[i] > r4_aligned[i] and rsi_7[i] > 55.0 and htf_bull
        breakout_bear = close[i] < s4_aligned[i] and rsi_7[i] < 45.0 and htf_bear
        
        if breakout_bull and desired_signal == 0.0:
            desired_signal = SIZE * 0.8
        elif breakout_bear and desired_signal == 0.0:
            desired_signal = -SIZE * 0.8
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (Reduce to half at 2R) ===
        if in_position and desired_signal != 0.0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    desired_signal = SIZE_HALF  # Reduce to half
            elif position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    desired_signal = -SIZE_HALF  # Reduce to half
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals