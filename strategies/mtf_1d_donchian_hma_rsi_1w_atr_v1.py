#!/usr/bin/env python3
"""
Experiment #1337: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Momentum

Hypothesis: Daily timeframe with weekly trend filter reduces noise while capturing
major moves. Donchian(20) breakouts capture momentum, 1w HMA(21) provides macro bias.
RSI(14) confirms momentum without being too restrictive. Wider entry bands ensure
sufficient trades (target 30-50/year). ATR trailing stop manages risk.

Key design choices:
1. 1w HMA(21) for macro trend - slower, more stable than 12h/4h
2. Donchian(20) breakout as primary trigger - proven momentum signal
3. RSI(14) with wide bands (25-75) - confirms without filtering too much
4. Mean reversion entry when price deviates >2*ATR from HMA
5. Simple ATR trailing stop (3x) - clean exit logic
6. Position size 0.30 - conservative for daily moves

Target: 30-50 trades/year, Sharpe > 0.612, trades >= 30 train, >= 5 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_atr_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period=50):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
    # Daily HMA for local trend
    hma_1d = calculate_hma(close, period=21)
    
    # HMA slope for trend confirmation
    hma_1d_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(hma_1d[i]) and not np.isnan(hma_1d[i-1]):
            hma_1d_slope[i] = hma_1d[i] - hma_1d[i-1]
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === LOCAL TREND (1d HMA) ===
        hma_bull = (close[i] > hma_1d[i]) and (hma_1d_slope[i] > 0)
        hma_bear = (close[i] < hma_1d[i]) and (hma_1d_slope[i] < 0)
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === RSI MOMENTUM ===
        rsi_bull = rsi[i] > 45.0
        rsi_bear = rsi[i] < 55.0
        rsi_strong_bull = rsi[i] > 55.0
        rsi_strong_bear = rsi[i] < 45.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === DISTANCE FROM HMA (mean reversion signal) ===
        hma_dist_pct = (close[i] - hma_1d[i]) / hma_1d[i] if hma_1d[i] > 0 else 0
        deep_oversold = hma_dist_pct < -0.05  # 5% below HMA
        deep_overbought = hma_dist_pct > 0.05  # 5% above HMA
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Multiple confluence paths (ensure trades happen)
        if macro_bull:
            # Path 1: Donchian breakout with RSI confirmation
            if breakout_long and rsi_bull:
                desired_signal = BASE_SIZE
            # Path 2: HMA bull + SMA50 bull + RSI strong
            elif hma_bull and above_sma50 and rsi_strong_bull:
                desired_signal = BASE_SIZE
            # Path 3: Mean reversion - deep oversold in macro bull
            elif deep_oversold and rsi[i] < 40.0:
                desired_signal = BASE_SIZE * 0.5
            # Path 4: Simple trend follow - price above both HMA and SMA50
            elif close[i] > hma_1d[i] and close[i] > sma_50[i] and rsi[i] > 50.0:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Multiple confluence paths
        elif macro_bear:
            # Path 1: Donchian breakout with RSI confirmation
            if breakout_short and rsi_bear:
                desired_signal = -BASE_SIZE
            # Path 2: HMA bear + SMA50 bear + RSI strong
            elif hma_bear and below_sma50 and rsi_strong_bear:
                desired_signal = -BASE_SIZE
            # Path 3: Mean reversion - deep overbought in macro bear
            elif deep_overbought and rsi[i] > 60.0:
                desired_signal = -BASE_SIZE * 0.5
            # Path 4: Simple trend follow - price below both HMA and SMA50
            elif close[i] < hma_1d[i] and close[i] < sma_50[i] and rsi[i] < 50.0:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
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