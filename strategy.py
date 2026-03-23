#!/usr/bin/env python3
"""
Experiment #1385: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: Previous 1h failures (#1375, #1380) used pullback entries with session filters
which generated too many low-quality trades. This strategy uses Connors RSI for mean
reversion entries BUT ONLY in direction of HTF trend (4h HMA + 1d HMA).

Key insight: Mean reversion works well in bear/range markets (2025 test period), but
pure mean reversion fails in strong trends. By filtering with 4h/1d HMA trend, we only
take mean reversion trades that align with higher timeframe direction.

Design:
1. 1d HMA(21) = macro bias (long only if price > 1d HMA, short only if price < 1d HMA)
2. 4h HMA(21) = intermediate trend confirmation (must align with 1d)
3. 1h Connors RSI < 15 or > 85 = extreme mean reversion entry
4. Volume > 0.8x 20-bar avg = confirmation (avoid low liquidity entries)
5. Choppiness(14) < 50 = avoid dead ranges (only trade when some movement exists)
6. ATR(14) 2.5x trailing stop = risk management
7. Position size 0.22 = conservative for 1h volatility
8. Entry cooldown 12 bars = prevent overtrading

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_meanreversion_4h1d_hma_trend_filter_v1"
timeframe = "1h"
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

def calculate_streak_rsi(close, period=2):
    """Streak RSI component of Connors RSI - measures consecutive up/down days"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(period, n):
        window = streak[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            avg_gain = np.mean(window[window > 0]) if np.any(window > 0) else 0
            avg_loss = abs(np.mean(window[window < 0])) if np.any(window < 0) else 0.001
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank - percentage of values in lookback period below current value"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    percent_rank = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < window[-1])
            percent_rank[i] = (count_below / (period - 1)) * 100.0
    
    return percent_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_streak_rsi(close, period=streak_period)
    percent_rank = calculate_percent_rank(close, period=pr_period)
    
    crsi = np.full(len(close), np.nan)
    mask = ~np.isnan(rsi_3) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_3[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        if highest > lowest and atr_sum > 0:
            chop[i] = 100.0 * np.log10((highest - lowest) / atr_sum) / np.log10(period)
    
    return chop

def calculate_volume_avg(volume, period=20):
    """Rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend filters
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    volume_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    last_entry_bar = -100  # Cooldown tracking
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        if np.isnan(volume_avg[i]) or volume_avg[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d HMA) - must align with entry direction ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - must align with 1d ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === TREND CONFLUENCE - both 1d and 4h must agree ===
        strong_bull_trend = macro_bull and trend_4h_bull
        strong_bear_trend = macro_bear and trend_4h_bear
        
        # === CHOPPINESS FILTER - avoid dead ranges ===
        trending_market = choppiness[i] < 50.0  # < 50 = some trend exists
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * volume_avg[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        
        # === ENTRY COOLDOWN (prevent overtrading) ===
        cooldown_ok = (i - last_entry_bar) >= 12
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bullish + CRSI oversold + volume + trending + cooldown
        if strong_bull_trend and crsi_oversold and volume_ok and trending_market and cooldown_ok:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRY: HTF bearish + CRSI overbought + volume + trending + cooldown
        elif strong_bear_trend and crsi_overbought and volume_ok and trending_market and cooldown_ok:
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
        
        # === CRSI EXIT (mean reversion complete) ===
        crsi_exit = False
        if in_position and position_side > 0 and crsi[i] > 55.0:
            crsi_exit = True  # Long exit when CRSI recovers above 55
        if in_position and position_side < 0 and crsi[i] < 45.0:
            crsi_exit = True  # Short exit when CRSI falls below 45
        
        if crsi_exit:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if abs(desired_signal) >= BASE_SIZE * 0.5:
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
                last_entry_bar = i
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                last_entry_bar = i
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