#!/usr/bin/env python3
"""
Experiment #835: 6h Primary + 12h/1d HTF — KAMA Adaptive Trend with RSI Divergence

Hypothesis: 6h timeframe captures multi-day swings better than 4h (too noisy) or 12h (too slow).
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency ratio - smooth in trends,
responsive in ranges. Combined with RSI(7) divergence detection for early reversal signals.

Key innovations:
1. KAMA(10,2,30) on 6h - adapts to market noise better than HMA/EMA
2. RSI(7) divergence: price LL + RSI HL = long, price HH + RSI LH = short
3. 12h KAMA for intermediate trend confirmation
4. 1d KAMA for long-term bias (only trade with HTF direction)
5. ATR-based dynamic sizing: reduce position when vol spikes (ATR ratio > 1.5)
6. Loose entry thresholds to ensure ≥30 trades/train, ≥3/test

Entry conditions (designed for trade generation):
- LONG: 1d KAMA bull + 12h KAMA bull + (RSI<40 OR RSI bullish divergence)
- SHORT: 1d KAMA bear + 12h KAMA bear + (RSI>60 OR RSI bearish divergence)

Target: Sharpe>0.45 (beat current best 0.424), trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete, reduced to 0.20 when ATR spikes
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_rsi_div_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    ER = |net change| / sum of absolute changes
    High ER (trending) = more responsive, Low ER (ranging) = smoother
    """
    n = len(close)
    if n < slow_period + period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(slow_period, n):
        net_change = abs(close[i] - close[i - slow_period])
        sum_changes = 0.0
        for j in range(1, slow_period + 1):
            sum_changes += abs(close[i - j + 1] - close[i - j])
        
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA with SMA of first period values
    kama[slow_period] = np.mean(close[slow_period - period + 1:slow_period + 1])
    
    for i in range(slow_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def detect_rsi_divergence(prices, rsi, lookback=5):
    """
    Detect RSI divergence signals
    Bullish: price makes lower low, RSI makes higher low
    Bearish: price makes higher high, RSI makes lower high
    Returns: div_signal (1=bullish, -1=bearish, 0=none)
    """
    n = len(prices)
    div_signal = np.zeros(n)
    low = prices['low'].values
    high = prices['high'].values
    
    for i in range(lookback + 2, n):
        if np.isnan(rsi[i]) or np.isnan(rsi[i - lookback]):
            continue
        
        # Check for bullish divergence (price LL, RSI HL)
        price_ll = low[i] < min(low[i-lookback:i])
        rsi_hl = rsi[i] > min(rsi[i-lookback:i])
        
        # Check for bearish divergence (price HH, RSI LH)
        price_hh = high[i] > max(high[i-lookback:i])
        rsi_lh = rsi[i] < max(rsi[i-lookback:i])
        
        if price_ll and rsi_hl and rsi[i] < 45:
            div_signal[i] = 1.0  # Bullish divergence
        elif price_hh and rsi_lh and rsi[i] > 55:
            div_signal[i] = -1.0  # Bearish divergence
    
    return div_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 6h
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # Detect RSI divergence
    rsi_div = detect_rsi_divergence(prices, rsi_7, lookback=5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_REDUCED = 0.20  # When vol spikes
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_6h[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 12h KAMA) ===
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        
        htf_12h_bull = close[i] > kama_12h_aligned[i]
        htf_12h_bear = close[i] < kama_12h_aligned[i]
        
        # === 6h KAMA TREND ===
        kama_6h_bull = close[i] > kama_6h[i]
        kama_6h_bear = close[i] < kama_6h[i]
        
        # === RSI CONDITIONS (loose for trade generation) ===
        rsi_oversold = rsi_7[i] < 40.0
        rsi_overbought = rsi_7[i] > 60.0
        rsi_extreme_oversold = rsi_7[i] < 30.0
        rsi_extreme_overbought = rsi_7[i] > 70.0
        
        # === RSI DIVERGENCE ===
        bullish_div = rsi_div[i] == 1.0
        bearish_div = rsi_div[i] == -1.0
        
        # === VOLATILITY ADJUSTMENT ===
        atr_ratio = atr_14[i] / atr_30[i] if not np.isnan(atr_30[i]) and atr_30[i] > 1e-10 else 1.0
        vol_spike = atr_ratio > 1.5
        
        # Select position size based on volatility
        current_size_base = SIZE_REDUCED if vol_spike else SIZE_BASE
        current_size_strong = SIZE_REDUCED if vol_spike else SIZE_STRONG
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + (RSI oversold OR divergence OR KAMA bull)
        if htf_1d_bull and htf_12h_bull:
            if rsi_oversold or bullish_div or kama_6h_bull:
                if rsi_extreme_oversold or bullish_div:
                    desired_signal = current_size_strong
                else:
                    desired_signal = current_size_base
        
        # SHORT: HTF bear + (RSI overbought OR divergence OR KAMA bear)
        elif htf_1d_bear and htf_12h_bear:
            if rsi_overbought or bearish_div or kama_6h_bear:
                if rsi_extreme_overbought or bearish_div:
                    desired_signal = -current_size_strong
                else:
                    desired_signal = -current_size_base
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= current_size_strong * 0.9:
            final_signal = current_size_strong
        elif desired_signal <= -current_size_strong * 0.9:
            final_signal = -current_size_strong
        elif desired_signal >= current_size_base * 0.9:
            final_signal = current_size_base
        elif desired_signal <= -current_size_base * 0.9:
            final_signal = -current_size_base
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals