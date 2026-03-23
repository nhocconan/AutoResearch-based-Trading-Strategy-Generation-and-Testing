#!/usr/bin/env python3
"""
Experiment #1381: 4h Primary + 1d HTF — Clean Trend Following with Fisher Transform Entry

Hypothesis: Previous regime-filter strategies (#1369, #1371, #1372, #1373) failed due to
over-filtering. Working strategies (#1374, #1376) used clean trend following with
multiple entry paths. Adding Fisher Transform improves entry timing in bear/range markets
by catching reversals that pure breakout strategies miss.

Key insight: 4h timeframe balances noise reduction with trade frequency. Fisher Transform
(period=9) identifies extreme reversals (crossing ±1.5) that precede trend moves.
Combined with Donchian breakout + HMA trend = high-probability entries.

Design:
1. 1d HMA(21) = macro trend bias (soft filter, not hard requirement)
2. 4h HMA(21) + slope = primary trend confirmation
3. Donchian(20) breakout = entry trigger
4. Fisher Transform(9) = entry timing refinement (catches reversals)
5. RSI(14) wide bands (30-70) = momentum confirmation without over-filtering
6. ATR(14) trailing stop 2.5x = risk management
7. Position size 0.30 = conservative for 4h volatility
8. FIVE entry paths per direction = ensures >=30 trades/train

Target: 30-50 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_donchian_hma_1d_rsi_atr_multipath_v1"
timeframe = "4h"
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

def calculate_hma_slope(hma, lookback=5):
    """HMA slope - positive = uptrend, negative = downtrend"""
    n = len(hma)
    slope = np.full(n, np.nan)
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i - lookback]):
            if hma[i - lookback] != 0:
                slope[i] = (hma[i] - hma[i - lookback]) / hma[i - lookback] * 100.0
    return slope

def calculate_rsi(close, period=14):
    """Relative Strength Index - wide bands for entry confirmation"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_fisher_transform(high, low, close, period=9):
    """Ehlers Fisher Transform - catches reversals in bear/range markets
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Calculate typical price
        typical = (high[i] + low[i] + close[i]) / 3.0
        
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (typical - lowest) / (highest - lowest) - 1.0
        
        # Clamp to avoid division issues
        normalized = max(-0.999, min(0.999, normalized))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i >= 1 and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_slope = calculate_hma_slope(hma_4h, lookback=5)
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
            continue
        if np.isnan(donchian_20_upper[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_4h[i]) or np.isnan(hma_4h_slope[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d HMA) - soft filter only ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA + slope) ===
        trend_bull = close[i] > hma_4h[i] and hma_4h_slope[i] > 0.0
        trend_bear = close[i] < hma_4h[i] and hma_4h_slope[i] < 0.0
        
        # === RSI MOMENTUM (WIDE bands to ensure trades) ===
        rsi_bull = rsi[i] > 30.0
        rsi_bear = rsi[i] < 70.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_20_upper[i-1]
        breakout_short = close[i] < donchian_20_lower[i-1]
        
        # === FISHER TRANSFORM REVERSAL ===
        fisher_bull = False
        fisher_bear = False
        
        if not np.isnan(fisher[i]) and not np.isnan(fisher_prev[i]):
            # Long: Fisher crosses above -1.5 from below
            if fisher_prev[i] < -1.5 and fisher[i] >= -1.5:
                fisher_bull = True
            # Short: Fisher crosses below +1.5 from above
            if fisher_prev[i] > 1.5 and fisher[i] <= 1.5:
                fisher_bear = True
        
        # === DESIRED SIGNAL - FIVE ENTRY PATHS PER DIRECTION ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS (any one triggers entry)
        # Path 1: Donchian breakout + trend confirmation (primary)
        if breakout_long and trend_bull and rsi_bull:
            desired_signal = BASE_SIZE
        # Path 2: Donchian breakout + macro confirmation (strong)
        elif breakout_long and macro_bull and rsi_strong_bull:
            desired_signal = BASE_SIZE
        # Path 3: Fisher reversal + above 4h HMA (reversal play)
        elif fisher_bull and close[i] > hma_4h[i]:
            desired_signal = BASE_SIZE * 0.5
        # Path 4: Price above both HMAs + positive slope (trend continuation)
        elif close[i] > hma_4h[i] and close[i] > hma_1d_aligned[i] and hma_4h_slope[i] > 0.05:
            desired_signal = BASE_SIZE * 0.5
        # Path 5: Strong RSI + trend (momentum play)
        elif rsi[i] > 55.0 and trend_bull:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY PATHS (any one triggers entry)
        # Path 1: Donchian breakout + trend confirmation (primary)
        elif breakout_short and trend_bear and rsi_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Donchian breakout + macro confirmation (strong)
        elif breakout_short and macro_bear and rsi_strong_bear:
            desired_signal = -BASE_SIZE
        # Path 3: Fisher reversal + below 4h HMA (reversal play)
        elif fisher_bear and close[i] < hma_4h[i]:
            desired_signal = -BASE_SIZE * 0.5
        # Path 4: Price below both HMAs + negative slope (trend continuation)
        elif close[i] < hma_4h[i] and close[i] < hma_1d_aligned[i] and hma_4h_slope[i] < -0.05:
            desired_signal = -BASE_SIZE * 0.5
        # Path 5: Weak RSI + trend (momentum play)
        elif rsi[i] < 45.0 and trend_bear:
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