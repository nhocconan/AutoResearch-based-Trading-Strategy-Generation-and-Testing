#!/usr/bin/env python3
"""
Experiment #1408: 30m Primary + 4h/1d HTF — Strict Confluence Pullback Strategy

Hypothesis: 30m timeframe with DUAL HTF confirmation (4h + 1d HMA) will provide
better trend filtering while using 30m only for precise pullback entries.
Key insight from failures: lower TF strategies fail due to TOO MANY TRADES.

Design:
1. 1d HMA(21) = macro trend direction (MUST align)
2. 4h HMA(21) = intermediate confirmation (MUST align with 1d)
3. 30m RSI(14) pullback = entry timing (30-45 for long, 55-70 for short)
4. 30m Bollinger position = entry quality (lower half for long, upper for short)
5. Volume filter (>1.0x 20-bar avg) = confirm genuine interest
6. Session filter (6-22 UTC) = avoid low-liquidity hours, reduce trades
7. ATR(14) trailing stop 2.5x = risk management
8. Position size 0.20 = conservative for 30m volatility (smaller than 4h/1d)

CRITICAL: Only enter when BOTH HTF trends align + RSI pullback + volume + session.
This should generate 40-80 trades/year (within limit for 30m).

Target: Sharpe > 0.618 (beat 1d baseline), trades >= 30 train, >= 3 test, DD > -50%
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_dual_hma_rsi_pullback_vol_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - for pullback detection"""
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
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands - for entry quality filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def extract_hour(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    if isinstance(open_time, np.ndarray):
        return (open_time // 3600000) % 24
    else:
        return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (30m) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Volume filter: 20-bar moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # Session hours (6-22 UTC = active trading hours)
    hours = extract_hour(open_time)
    in_session = (hours >= 6) & (hours <= 22)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Conservative for 30m
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - MUST align ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - MUST align with 1d ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === BOTH HTF MUST AGREE (strict confluence) ===
        trend_aligned_long = macro_bull and inter_bull
        trend_aligned_short = macro_bear and inter_bear
        
        # === RSI PULLBACK (wider bands to ensure trades) ===
        # Long: RSI pulled back to 30-50 zone in uptrend
        rsi_pullback_long = 30.0 <= rsi[i] <= 50.0
        # Short: RSI rallied to 50-70 zone in downtrend
        rsi_pullback_short = 50.0 <= rsi[i] <= 70.0
        
        # === BOLLINGER POSITION (entry quality) ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_range
        else:
            bb_position = 0.5
        
        bb_lower_half = bb_position < 0.5  # Good for long entries
        bb_upper_half = bb_position > 0.5  # Good for short entries
        
        # === VOLUME FILTER (confirm interest) ===
        volume_confirmed = vol_ratio[i] >= 0.8  # At least 80% of average
        
        # === SESSION FILTER (active hours only) ===
        session_active = in_session[i]
        
        # === PRICE MOMENTUM (break of previous bar) ===
        prev_high = high[i-1] if i > 0 else close[i]
        prev_low = low[i-1] if i > 0 else close[i]
        
        momentum_long = close[i] > prev_high
        momentum_short = close[i] < prev_low
        
        # === DESIRED SIGNAL - STRICT CONFLUENCE ===
        desired_signal = 0.0
        
        # LONG: HTF aligned + RSI pullback + BB lower half + volume + session + momentum
        if trend_aligned_long and rsi_pullback_long and bb_lower_half and volume_confirmed and session_active:
            if momentum_long:
                desired_signal = BASE_SIZE
            else:
                # Enter on pullback without momentum break (aggressive)
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT: HTF aligned + RSI pullback + BB upper half + volume + session + momentum
        elif trend_aligned_short and rsi_pullback_short and bb_upper_half and volume_confirmed and session_active:
            if momentum_short:
                desired_signal = -BASE_SIZE
            else:
                # Enter on pullback without momentum break (aggressive)
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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