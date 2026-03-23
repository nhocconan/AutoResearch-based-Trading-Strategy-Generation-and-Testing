#!/usr/bin/env python3
"""
Experiment #1283: 1d Primary + 1w HTF — Simplified Dual Regime with LOOSE Entries

Hypothesis: Recent failures (#1272-1282) all have Sharpe=0.000 = ZERO TRADES.
Entry conditions were TOO STRICT with multiple confluence requirements.

This strategy SIMPLIFIES to ensure trades happen:
1. 1w HMA(21) for macro trend direction (slow, stable filter)
2. 1d HMA(16/48) crossover for entry timing
3. Choppiness Index regime (CHOP>50=range, CHOP<50=trend) - WIDER bands
4. RSI(14) with LOOSE thresholds (25-75 vs strict 30-70)
5. ATR(14) 2.5x trailing stop for risk management
6. Position size: 0.28 (discrete levels)

Key changes from failed experiments:
- REMOVED ADX filter (was blocking too many signals)
- REMOVED Bollinger Band confluence (was too restrictive)
- WIDENED RSI bands (25-75 vs 30-70 or 20-80)
- WIDENED Choppiness threshold (50 vs 45-55 buffer)
- SIMPLIFIED regime logic (fewer conditions to satisfy)
- No hysteresis buffer on signal changes

Target: Sharpe > 0.612, trades >= 80 train (20+/year), >= 12 test (3+/year)
Timeframe: 1d (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_chop_1w_regime_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection
    CHOP > 50 = ranging (mean revert)
    CHOP < 50 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average - long-term trend filter"""
    n = len(close)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi := chop[i]) or np.isnan(rsi[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        in_range = chop[i] > 50.0  # Ranging market
        in_trend = chop[i] <= 50.0  # Trending market
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === LONG-TERM TREND (SMA200) ===
        long_bull = close[i] > sma_200[i]
        long_bear = close[i] < sma_200[i]
        
        # === LOCAL TREND (1d HMA crossover) ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # === MOMENTUM (RSI) - LOOSE thresholds ===
        rsi_oversold = rsi[i] < 45.0  # Wider than typical 30
        rsi_overbought = rsi[i] > 55.0  # Wider than typical 70
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow trend with HMA crossover
        if in_trend:
            # Long: Macro bull + HMA bull + RSI not overbought
            if macro_bull and hma_bull and rsi[i] < 70.0:
                desired_signal = BASE_SIZE
            # Short: Macro bear + HMA bear + RSI not oversold
            elif macro_bear and hma_bear and rsi[i] > 30.0:
                desired_signal = -BASE_SIZE
        
        # RANGING REGIME: Mean revert with RSI extremes
        elif in_range:
            # Long: RSI oversold + price above SMA200 (bullish bias)
            if rsi[i] < 35.0 and long_bull:
                desired_signal = BASE_SIZE
            # Short: RSI overbought + price below SMA200 (bearish bias)
            elif rsi[i] > 65.0 and long_bear:
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
        
        # === OUTPUT SIGNAL ===
        final_signal = desired_signal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if final_signal > 0.1:
            final_signal = BASE_SIZE
        elif final_signal < -0.1:
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