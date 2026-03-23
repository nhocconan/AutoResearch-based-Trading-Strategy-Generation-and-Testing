#!/usr/bin/env python3
"""
Experiment #1320: 1h Primary + 4h/12h HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Lower TF (1h) failed in #1315 (Sharpe -5.330) due to TOO MANY filters
(volume + session + multiple regime). This strategy SIMPLIFIES:
1. 12h HMA for macro trend (bull/bear regime)
2. 4h HMA for intermediate trend confirmation
3. 1h RSI(14) pullback entries with WIDE bands (30-60 long, 40-70 short)
4. ATR(14) trailing stop at 2.5x for risk management
5. NO session filter (kills trades)
6. NO volume filter (kills trades)
7. LOOSE confluence: only need 2/3 HTF signals aligned

Key insight from #1315 failure: adding filters REDUCED trades to near-zero.
This uses MINIMAL filters to ensure trade generation while keeping HTF direction.

Target: 40-80 trades/year on 1h, Sharpe > 0.612, trades >= 40 train, >= 5 test
Timeframe: 1h
Size: 0.25 discrete levels (smaller for lower TF to reduce fee impact)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_trend_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

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

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 4h SMA50 for additional macro filter
    sma_4h_raw = calculate_sma(df_4h['close'].values, period=50)
    sma_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_4h_raw)
    
    # Calculate primary (1h) indicators
    hma_fast = calculate_hma(close, period=13)
    hma_slow = calculate_hma(close, period=34)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA + SMA50) ===
        # Need at least 1 of 2 bullish for long bias
        h4_bull_count = 0
        if close[i] > hma_4h_aligned[i]:
            h4_bull_count += 1
        if close[i] > sma_4h_aligned[i]:
            h4_bull_count += 1
        
        h4_bear_count = 0
        if close[i] < hma_4h_aligned[i]:
            h4_bear_count += 1
        if close[i] < sma_4h_aligned[i]:
            h4_bear_count += 1
        
        # === LOCAL TREND (1h HMA crossover) ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Macro bull + (4h bull OR local bull) + RSI pullback
        # WIDE RSI bands to ensure trades happen
        if macro_bull:
            # Need at least 1 bullish signal from 4h or 1h
            if h4_bull_count >= 1 or hma_bull:
                # RSI pullback in uptrend (30-55 range - WIDE)
                if 30.0 <= rsi[i] <= 55.0:
                    desired_signal = BASE_SIZE
                # RSI breaking above 45 with momentum
                elif 45.0 < rsi[i] < 60.0 and hma_bull:
                    desired_signal = BASE_SIZE
                # RSI oversold bounce with SMA200 support
                elif rsi[i] < 35.0 and above_sma200:
                    desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Macro bear + (4h bear OR local bear) + RSI bounce
        elif macro_bear:
            # Need at least 1 bearish signal from 4h or 1h
            if h4_bear_count >= 1 or hma_bear:
                # RSI bounce in downtrend (45-70 range - WIDE)
                if 45.0 <= rsi[i] <= 70.0:
                    desired_signal = -BASE_SIZE
                # RSI breaking below 55 with momentum
                elif 40.0 < rsi[i] < 55.0 and hma_bear:
                    desired_signal = -BASE_SIZE
                # RSI overbought rejection with SMA200 resistance
                elif rsi[i] > 65.0 and below_sma200:
                    desired_signal = -BASE_SIZE
        
        # === RANGE MARKET: Mean revert at extremes (when HTF unclear) ===
        # Only if 12h HMA is flat (price within 2% of HMA)
        hma_12h_flat = abs(close[i] - hma_12h_aligned[i]) / hma_12h_aligned[i] < 0.02
        if hma_12h_flat and desired_signal == 0.0:
            # Long at RSI oversold
            if rsi[i] < 28.0:
                desired_signal = BASE_SIZE
            # Short at RSI overbought
            elif rsi[i] > 72.0:
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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