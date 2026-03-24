#!/usr/bin/env python3
"""
Experiment #1525: 1h Primary + 4h/1d HTF — KAMA Trend + RSI Extremes + ATR Stop

Hypothesis: Based on #1515/#1518 failures (0 trades from session/volume filters),
this strategy removes ALL session and volume filters. Key insights:
1. Session filters killed trade generation (#1515, #1518 both Sharpe=0.000)
2. Volume filters add complexity without edge (#1517, #1520 negative Sharpe)
3. 1h needs VERY strict HTF alignment (4h + 1d must agree) to limit trades
4. RSI(7) extremes (≤30 / ≥70) generate more trades than pullback ranges
5. KAMA adapts better to crypto regime changes than HMA/EMA

Design:
- 4h KAMA(21) for intermediate trend bias (HTF filter 1)
- 1d KAMA(21) for macro trend bias (HTF filter 2 - BOTH must agree)
- 1h RSI(7) for entry timing (extremes: ≤30 long, ≥70 short)
- ATR(14) 2.0x trailing stop for tight risk management
- Position size 0.25 (smaller for 1h to reduce fee impact)
- Target: 40-80 trades/train (4 years), 10-20 trades/test (15 months)

Timeframe: 1h (as required by experiment)
HTF: 4h + 1d (both must agree for trade direction)
Position Size: 0.25 (discrete: 0.0, ±0.25)
Target: Sharpe > 0.618 (beat current best), DD < -30%

CRITICAL CHANGES FROM FAILED #1515:
- NO session filter (8-20 UTC killed trades)
- NO volume filter (added complexity, no edge)
- RSI extremes (30/70) not pullback ranges (40-60 too narrow)
- Both 4h AND 1d must agree (stricter HTF = fewer but better trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_rsi7_4h1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency/volatility
    KAMA[i] = KAMA[i-1] + SC * (price - KAMA[i-1])
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    ER = Change / Sum(|Change|) over er_period
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close[er_period:] - close[:-er_period])
    sum_abs_change = np.zeros(len(change))
    for i in range(len(change)):
        sum_abs_change[i] = np.sum(np.abs(np.diff(close[i:i+er_period+1])))
    
    er = np.zeros(len(change))
    mask = sum_abs_change > 1e-10
    er[mask] = change[mask] / sum_abs_change[mask]
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA at SMA of first er_period
    kama[er_period] = np.mean(close[:er_period+1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        er_idx = i - er_period
        if er_idx < len(er):
            sc = (er[er_idx] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for 1h entries"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h KAMA for intermediate trend
    kama_4h_raw = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_raw)
    
    # Calculate and align 1d KAMA for macro trend
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h (fee efficiency)
    
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
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_4h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND ALIGNMENT (BOTH 4h AND 1d MUST AGREE) ===
        # This is the key filter to limit trades on 1h
        htf_bullish = (close[i] > kama_4h_aligned[i]) and (close[i] > kama_1d_aligned[i])
        htf_bearish = (close[i] < kama_4h_aligned[i]) and (close[i] < kama_1d_aligned[i])
        
        # === RSI EXTREMES (wider bands for more trades) ===
        rsi_oversold = rsi[i] <= 30.0  # Long entry
        rsi_overbought = rsi[i] >= 70.0  # Short entry
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Both HTF bullish + RSI oversold extreme
        if htf_bullish and rsi_oversold:
            desired_signal = BASE_SIZE
        
        # SHORT: Both HTF bearish + RSI overbought extreme
        elif htf_bearish and rsi_overbought:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x - tighter for 1h) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
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