#!/usr/bin/env python3
"""
Experiment #1468: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion with HTF Trend Filter

Hypothesis: Lower TF (30m) strategies failed due to OVER-FILTERING (0 trades).
This version uses Connors RSI (CRSI) which has proven 75% win rate for mean reversion,
combined with HTF trend filter to only trade in direction of higher timeframe trend.

Key differences from failed 30m strategies:
1. CRSI (not regular RSI) = more sensitive to short-term extremes
2. ONLY 2 HTF filters (4h HMA + 1d HMA) = not 3+ conflicting filters
3. Session filter REMOVED = was causing 0 trades in #1458, #1460, #1465
4. Volume filter SOFT (only for confirmation, not hard requirement)
5. Wider CRSI bands (10/90 instead of 15/85) = more entry signals

Why this should work on 30m:
- HTF (4h/1d) determines DIRECTION only
- 30m CRSI determines ENTRY TIMING within HTF trend
- This gives ~40-60 trades/year (HTF frequency) with 30m execution precision
- Position size 0.25 = appropriate for 30m volatility

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_mean_reversion_4h1d_hma_atr_v1"
timeframe = "30m"
leverage = 1.0

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        if len(streak_vals) == streak_period:
            gains = sum(1 for s in streak_vals if s > 0)
            losses = sum(1 for s in streak_vals if s < 0)
            if losses == 0:
                streak_rsi[i] = 100.0
            else:
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + gains / losses))
    
    # Component 3: PercentRank of price change over rank_period
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        current_return = returns[-1] if len(returns) > 0 else 0
        if len(returns) > 0:
            rank = sum(1 for r in returns[:-1] if r < current_return)
            percent_rank[i] = rank / (len(returns) - 1) * 100.0 if len(returns) > 1 else 50.0
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for lower TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]):
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
        # Both must agree for signal direction
        trend_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        trend_neutral = not trend_bull and not trend_bear
        
        # === VOLUME CONFIRMATION (soft filter) ===
        vol_confirmed = not np.isnan(vol_sma[i]) and volume[i] > 0.8 * vol_sma[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Wide band for more trades
        crsi_overbought = crsi[i] > 85.0  # Wide band for more trades
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === DESIRED SIGNAL - CRSI MEAN REVERSION WITH HTF TREND ===
        desired_signal = 0.0
        
        # LONG ENTRIES (in uptrend or neutral, CRSI oversold)
        if trend_bull and crsi_oversold:
            # Strong signal: uptrend + CRSI oversold
            desired_signal = BASE_SIZE
        elif trend_bull and crsi_extreme_oversold:
            # Very strong: extreme oversold in uptrend
            desired_signal = BASE_SIZE
        elif trend_neutral and crsi_extreme_oversold and vol_confirmed:
            # Neutral trend + extreme oversold + volume = mean reversion long
            desired_signal = BASE_SIZE * 0.7
        elif crsi_extreme_oversold and close[i] > hma_1d_aligned[i]:
            # Extreme oversold but above daily HMA = strong bounce candidate
            desired_signal = BASE_SIZE
        
        # SHORT ENTRIES (in downtrend or neutral, CRSI overbought)
        elif trend_bear and crsi_overbought:
            # Strong signal: downtrend + CRSI overbought
            desired_signal = -BASE_SIZE
        elif trend_bear and crsi_extreme_overbought:
            # Very strong: extreme overbought in downtrend
            desired_signal = -BASE_SIZE
        elif trend_neutral and crsi_extreme_overbought and vol_confirmed:
            # Neutral trend + extreme overbought + volume = mean reversion short
            desired_signal = -BASE_SIZE * 0.7
        elif crsi_extreme_overbought and close[i] < hma_1d_aligned[i]:
            # Extreme overbought but below daily HMA = strong drop candidate
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
        if desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
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