#!/usr/bin/env python3
"""
Experiment #1438: 30m Primary + 4h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous 30m strategies failed due to TOO MANY filters (CRSI + CHOP + Volume + Session)
resulting in 0 trades. This strategy SIMPLIFIES entry conditions while keeping HTF trend filter.

Key changes from failed #1428, #1430, #1434, #1435:
1. Remove Choppiness Index (too restrictive, killed trades)
2. Remove Connors RSI (too complex, 0 trades on 30m)
3. Remove session filter (8-20 UTC too restrictive for crypto 24/7)
4. Remove volume filter (killed trade generation)
5. Use SIMPLE RSI(14) with wider thresholds (35/65 not 15/85)
6. Use 4h HMA for trend + 1d HMA for macro filter (dual HTF confirmation)

Design:
1. 4h HMA(21) = primary trend direction (call ONCE before loop)
2. 1d HMA(21) = macro trend filter (call ONCE before loop)
3. 30m RSI(14) = entry trigger (long <40, short >60) - RELAXED for trade generation
4. 30m price vs 30m SMA(50) = additional trend confirmation
5. ATR(14) trailing stop 2.5x = risk management
6. Position size 0.25 = conservative for 30m volatility

Why this should work:
- Fewer filters = MORE trades (critical for 30m to hit 30-80/year target)
- Dual HTF (4h + 1d) provides strong trend bias without over-complicating
- RSI(14) with 40/60 thresholds generates signals in both bull and bear markets
- 30m timeframe captures intraday moves while HTF filters major trend

Target: 40-80 trades/year, Sharpe > 0.618 (beat current best), trades >= 30 train, >= 5 test
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_dual_atr_v1"
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

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        if not np.any(np.isnan(window)):
            sma[i] = np.mean(window)
    
    return sma

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
    
    # Calculate and align 4h HMA for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    rsi_14 = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, period=50)
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(sma_50[i]):
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
        
        # === HTF TREND FILTERS (4h + 1d HMA) ===
        # Both must agree for strong signal
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong trend: both 4h and 1d agree
        strong_bull = trend_4h_bull and trend_1d_bull
        strong_bear = trend_4h_bear and trend_1d_bear
        
        # Weak trend: only 4h agrees (allows more trades)
        weak_bull = trend_4h_bull and not trend_1d_bear
        weak_bear = trend_4h_bear and not trend_1d_bull
        
        # === 30m LOCAL TREND (SMA50) ===
        local_bull = close[i] > sma_50[i]
        local_bear = close[i] < sma_50[i]
        
        # === RSI ENTRY TRIGGERS (RELAXED for trade generation) ===
        rsi_oversold = rsi_14[i] < 40.0  # Long entry (was 35, relaxed)
        rsi_overbought = rsi_14[i] > 60.0  # Short entry (was 65, relaxed)
        
        # RSI extreme for stronger signals
        rsi_deep_oversold = rsi_14[i] < 30.0
        rsi_deep_overbought = rsi_14[i] > 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # Path 1: Strong bull trend + RSI pullback (primary entry)
        if strong_bull and rsi_oversold and local_bull:
            desired_signal = BASE_SIZE
        # Path 2: Weak bull trend + deep RSI oversold (counter-trend in bull)
        elif weak_bull and rsi_deep_oversold:
            desired_signal = BASE_SIZE * 0.5
        # Path 3: Strong bull + RSI very deep oversold (panic buy)
        elif strong_bull and rsi_14[i] < 25.0:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        # Path 1: Strong bear trend + RSI pullback (primary entry)
        elif strong_bear and rsi_overbought and local_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Weak bear trend + deep RSI overbought (counter-trend in bear)
        elif weak_bear and rsi_deep_overbought:
            desired_signal = -BASE_SIZE * 0.5
        # Path 3: Strong bear + RSI very deep overbought (panic sell)
        elif strong_bear and rsi_14[i] > 75.0:
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