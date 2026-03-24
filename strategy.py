#!/usr/bin/env python3
"""
Experiment #043: 6h Primary + 1w/1d HTF — Weekly Pivot Bounce + RSI Divergence + Volume

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). After 7 failed 6h 
experiments using CHOP/CRSI/Fisher/Donchian patterns, I'm trying a DIFFERENT approach:

1. WEEKLY PIVOT LEVELS as major S/R zones (proven in traditional trading)
   - Classic Pivot: P = (H + L + C) / 3 from weekly bars
   - Support/Resistance: R1 = 2P - L, S1 = 2P - H
   - Price bouncing off weekly S1/S2 with RSI oversold = high-probability long
   - Price rejecting weekly R1/R2 with RSI overbought = high-probability short

2. RSI DIVERGENCE (not just levels) - more robust than absolute RSI
   - Bullish divergence: price makes lower low, RSI makes higher low
   - Bearish divergence: price makes higher high, RSI makes lower high
   - 3-bar lookback for divergence detection

3. VOLUME CONFIRMATION via taker_buy_volume ratio
   - Long: taker_buy_ratio > 0.55 (buying pressure) at support
   - Short: taker_buy_ratio < 0.45 (selling pressure) at resistance

4. VOLATILITY-ADJUSTED POSITION SIZING
   - Base size 0.28, reduce to 0.20 when ATR(14) > 1.5x ATR(50)
   - Prevents oversized positions during vol spikes

5. HTF BIAS from 1d HMA(50) - only trade in direction of daily trend
   - Reduces counter-trend trades that fail in strong trends

Why this might work on 6h:
- Weekly pivots are MAJOR levels that institutions watch
- 6h bars give enough resolution to see bounces/rejections clearly
- Divergence + volume filter reduces false signals
- Vol-adjusted sizing controls drawdown during 2022-style crashes

Target: 30-60 trades/year, Sharpe > 0.167 (beat current best), DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_rsi_div_vol_1w1d_v1"
timeframe = "6h"
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
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_weekly_pivots(df_1w):
    """
    Calculate Classic Pivot Points from weekly data
    P = (H + L + C) / 3
    R1 = 2*P - L, S1 = 2*P - H
    R2 = P + (H - L), S2 = P - (H - L)
    """
    n = len(df_1w)
    pivots = np.zeros((n, 5))  # S2, S1, P, R1, R2
    pivots[:] = np.nan
    
    for i in range(n):
        h = df_1w['high'].iloc[i]
        l = df_1w['low'].iloc[i]
        c = df_1w['close'].iloc[i]
        
        p = (h + l + c) / 3.0
        r1 = 2.0 * p - l
        s1 = 2.0 * p - h
        r2 = p + (h - l)
        s2 = p - (h - l)
        
        pivots[i] = [s2, s1, p, r1, r2]
    
    return pivots

def detect_rsi_divergence(close, rsi, lookback=3):
    """
    Detect RSI divergence
    Returns: 1 = bullish div, -1 = bearish div, 0 = none
    Bullish: price lower low, RSI higher low
    Bearish: price higher high, RSI lower high
    """
    n = len(close)
    div = np.zeros(n)
    
    for i in range(lookback + 2, n):
        # Check for bullish divergence
        price_ll = close[i] < min(close[i-lookback:i])
        rsi_hl = rsi[i] > min(rsi[i-lookback:i])
        if price_ll and rsi_hl and not np.isnan(rsi[i]):
            div[i] = 1
        
        # Check for bearish divergence
        price_hh = close[i] > max(close[i-lookback:i])
        rsi_lh = rsi[i] < max(rsi[i-lookback:i])
        if price_hh and rsi_lh and not np.isnan(rsi[i]):
            div[i] = -1 if div[i] == 0 else div[i]  # don't overwrite bullish
    
    return div

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivots and align to 6h
    weekly_pivots_raw = calculate_weekly_pivots(df_1w)
    # Align each pivot level separately
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 0])
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 1])
    p_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 2])
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 3])
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 4])
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    hma_6h = calculate_hma(close, period=21)
    rsi_div = detect_rsi_divergence(close, rsi, lookback=3)
    
    # Taker buy ratio (volume sentiment)
    taker_ratio = np.zeros(n)
    taker_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 1e-10:
            taker_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% base position size
    MIN_SIZE = 0.20   # Reduced size in high vol
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA50) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY-ADJUSTED SIZING ===
        vol_ratio = atr[i] / (atr_50[i] + 1e-10) if not np.isnan(atr_50[i]) else 1.0
        position_size = BASE_SIZE if vol_ratio < 1.5 else MIN_SIZE
        
        # === WEEKLY PIVOT ZONES ===
        # Define tolerance as 0.5% of price for "near pivot"
        pivot_tolerance = close[i] * 0.005
        
        near_s2 = abs(close[i] - s2_aligned[i]) < pivot_tolerance
        near_s1 = abs(close[i] - s1_aligned[i]) < pivot_tolerance
        near_p = abs(close[i] - p_aligned[i]) < pivot_tolerance
        near_r1 = abs(close[i] - r1_aligned[i]) < pivot_tolerance
        near_r2 = abs(close[i] - r2_aligned[i]) < pivot_tolerance
        
        # Price below pivot = at support, above = at resistance
        at_support = close[i] <= p_aligned[i]
        at_resistance = close[i] >= p_aligned[i]
        
        # === VOLUME SENTIMENT ===
        buy_pressure = taker_ratio[i] > 0.55
        sell_pressure = taker_ratio[i] < 0.45
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_bull_div = rsi_div[i] == 1
        rsi_bear_div = rsi_div[i] == -1
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === ENTRY SIGNALS ===
        desired_signal = 0.0
        
        # LONG entries (at support zones)
        long_score = 0
        
        # Major support bounce: near S1/S2 + RSI oversold + HTF not bearish
        if (near_s1 or near_s2) and rsi_oversold and not htf_bear:
            long_score += 3
        # Pivot support + bullish divergence + buy pressure
        if at_support and rsi_bull_div and buy_pressure:
            long_score += 3
        # Near support + RSI oversold + volume confirmation + HMA bull
        if (near_s1 or near_s2 or near_p) and rsi_oversold and buy_pressure and hma_bull:
            long_score += 2
        # Divergence alone with HTF bull
        if rsi_bull_div and htf_bull and rsi_oversold:
            long_score += 2
        
        # SHORT entries (at resistance zones)
        short_score = 0
        
        # Major resistance rejection: near R1/R2 + RSI overbought + HTF not bullish
        if (near_r1 or near_r2) and rsi_overbought and not htf_bull:
            short_score += 3
        # Pivot resistance + bearish divergence + sell pressure
        if at_resistance and rsi_bear_div and sell_pressure:
            short_score += 3
        # Near resistance + RSI overbought + volume confirmation + HMA bear
        if (near_r1 or near_r2 or near_p) and rsi_overbought and sell_pressure and hma_bear:
            short_score += 2
        # Divergence alone with HTF bear
        if rsi_bear_div and htf_bear and rsi_overbought:
            short_score += 2
        
        # Generate signal based on scores
        if long_score >= 3 and short_score < 2:
            desired_signal = position_size
        elif short_score >= 3 and long_score < 2:
            desired_signal = -position_size
        elif long_score >= 2 and short_score == 0 and htf_bull:
            desired_signal = position_size * 0.7
        elif short_score >= 2 and long_score == 0 and htf_bear:
            desired_signal = -position_size * 0.7
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = MIN_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -MIN_SIZE
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