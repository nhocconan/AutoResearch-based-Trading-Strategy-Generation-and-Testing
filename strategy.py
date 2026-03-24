#!/usr/bin/env python3
"""
Experiment #1550: 1h Primary + 4h/12h HTF — Simplified Multi-TF Breakout

Hypothesis: After 11 failed experiments, the pattern is clear:
1. 1h strategies fail due to EITHER: too many trades (fee drag) OR too strict (0 trades)
2. #1543 (1d Donchian+HMA+RSI) worked with Sharpe=0.164 — simple logic wins
3. For 1h: MUST use 4h/12h for DIRECTION, 1h only for ENTRY TIMING
4. Avoid complex regime filters (CHOP, CRSI) — they caused 0 trades in #1545, #1547, #1548, #1549
5. Key: LOOSE entry conditions but HTF filter ensures quality

Strategy Design:
- HTF Bias: 4h HMA(21) + 12h HMA(21) — BOTH must agree for direction
- Entry: 1h Donchian(20) breakout in direction of HTF trend
- Filter: RSI(14) between 35-65 (avoid extremes, but not too strict)
- Stop: 2.5x ATR(14) trailing stop via signal→0
- Size: 0.25 (smaller for 1h to reduce fee impact)
- Target: 40-80 trades/year (strict enough to avoid whipsaw, loose enough to fire)

Why this should beat #1543 (Sharpe=0.618):
- 1h captures more moves than 1d while 4h/12h filter prevents noise
- Dual HTF agreement (4h AND 12h) ensures high-probability setups
- Donchian breakout works in both bull and bear markets
- Simpler than failed 1h attempts (#1540, #1545) — no session/volume filters that kill trades

Timeframe: 1h (required for this experiment)
HTF: 4h HMA + 12h HMA for dual confirmation
Target: Sharpe > 0.618, trades > 40/train, > 5/test, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_hma_4h12h_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout system
    Upper = highest high of last n periods
    Lower = lowest low of last n periods
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for long-term trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h to reduce fee impact
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h AND 12h must agree) ===
        # Bullish: price above BOTH 4h and 12h HMA
        htf_bull = (close[i] > hma_4h_aligned[i]) and (close[i] > hma_12h_aligned[i])
        # Bearish: price below BOTH 4h and 12h HMA
        htf_bear = (close[i] < hma_4h_aligned[i]) and (close[i] < hma_12h_aligned[i])
        
        # === DONCHIAN BREAKOUT (1h) ===
        # Use previous bar's Donchian levels to avoid look-ahead
        prev_upper = donchian_upper[i - 1] if i > 0 else np.nan
        prev_lower = donchian_lower[i - 1] if i > 0 else np.nan
        
        breakout_long = (not np.isnan(prev_upper)) and (close[i] > prev_upper)
        breakout_short = (not np.isnan(prev_lower)) and (close[i] < prev_lower)
        
        # === RSI FILTER (avoid extremes but not too strict) ===
        # Long: RSI not overbought (< 70)
        # Short: RSI not oversold (> 30)
        rsi_long_ok = rsi_14[i] < 70.0
        rsi_short_ok = rsi_14[i] > 30.0
        
        # === ENTRY LOGIC — HTF direction + 1h breakout + RSI filter ===
        desired_signal = 0.0
        
        # LONG: HTF bullish + Donchian breakout + RSI not overbought
        if htf_bull and breakout_long and rsi_long_ok:
            desired_signal = BASE_SIZE
        
        # SHORT: HTF bearish + Donchian breakout + RSI not oversold
        elif htf_bear and breakout_short and rsi_short_ok:
            desired_signal = -BASE_SIZE
        
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
                # Position flip
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