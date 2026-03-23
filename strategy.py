#!/usr/bin/env python3
"""
Experiment #1398: 30m Primary + 4h/1d HTF — Simplified Trend Pullback Strategy

Hypothesis: Previous 30m/1h failures (#1388, #1390, #1395 Sharpe=0.000) were caused by 
OVER-FILTERING with too many confluence conditions (Choppiness + CRSI + Session + Volume).
The working patterns (#1391, #1396) used SIMPLE trend following: HTF HMA + LTF RSI pullback.

Key insight from 1039 failed strategies: Lower TF needs FEWER filters, not more.
Each additional filter reduces trade frequency exponentially. For 30m to generate 30-80 
trades/year, we need 2-3 simple conditions, not 5-6 complex ones.

Design:
1. 4h HMA(21) = macro trend direction (proven on 4h/12h/1d)
2. 1d HMA(21) = higher-order trend confirmation (avoid counter-trend trades)
3. 30m RSI(14) pullback = entry timing (RSI 35-45 long, 55-65 short in trend)
4. 30m ATR(14) trailing stop 2.5x = risk management (proven)
5. Position size 0.25 = conservative for 30m volatility
6. NO session/volume/choppiness filters (these killed trade frequency)
7. Multiple RSI entry zones = ensures trade frequency

Target: 40-80 trades/year, Sharpe > 0.618 (beat 1d baseline), trades >= 30 train, >= 5 test
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_pullback_4h1d_rsi_atr_simple_v1"
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
    """Relative Strength Index - for pullback entry timing"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intermediate trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - highest order filter ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - confirmation ===
        intermediate_bull = close[i] > hma_4h_aligned[i]
        intermediate_bear = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK ZONES (entry timing) ===
        # Long: RSI pulled back to 35-50 in uptrend
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        rsi_strong_long = rsi[i] > 50.0
        rsi_oversold = rsi[i] < 40.0
        
        # Short: RSI pulled back to 50-65 in downtrend
        rsi_pullback_short = 50.0 <= rsi[i] <= 65.0
        rsi_strong_short = rsi[i] < 50.0
        rsi_overbought = rsi[i] > 60.0
        
        # === DESIRED SIGNAL - MULTIPLE ENTRY PATHS ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS (any one triggers entry)
        # Path 1: Both HTF trends bull + RSI pullback (highest probability)
        if macro_bull and intermediate_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # Path 2: 1d bull + RSI oversold (deep pullback entry)
        elif macro_bull and rsi_oversold:
            desired_signal = BASE_SIZE
        # Path 3: Both HTF bull + RSI recovering (momentum continuation)
        elif macro_bull and intermediate_bull and rsi_strong_long:
            desired_signal = BASE_SIZE * 0.5
        # Path 4: 4h bull alone + RSI neutral (simpler entry)
        elif intermediate_bull and 45.0 <= rsi[i] <= 55.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY PATHS (any one triggers entry)
        # Path 1: Both HTF trends bear + RSI pullback (highest probability)
        elif macro_bear and intermediate_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        # Path 2: 1d bear + RSI overbought (deep pullback entry)
        elif macro_bear and rsi_overbought:
            desired_signal = -BASE_SIZE
        # Path 3: Both HTF bear + RSI recovering (momentum continuation)
        elif macro_bear and intermediate_bear and rsi_strong_short:
            desired_signal = -BASE_SIZE * 0.5
        # Path 4: 4h bear alone + RSI neutral (simpler entry)
        elif intermediate_bear and 45.0 <= rsi[i] <= 55.0:
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
        if abs(desired_signal) >= BASE_SIZE * 0.4:
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