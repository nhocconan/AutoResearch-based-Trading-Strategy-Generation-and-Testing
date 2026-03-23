#!/usr/bin/env python3
"""
Experiment #1330: 1h Primary + 4h/12h HTF — Choppiness Regime + Connors RSI + HTF HMA Trend

Hypothesis: Bear/range markets (2022 crash, 2025 test) destroy pure trend strategies.
Choppiness Index detects regime: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow).
Connors RSI (CRSI) catches reversals with 75% win rate in ranges.
4h/12h HMA provides macro trend bias to avoid counter-trend trades.
Session filter (8-20 UTC) ensures liquidity. Volume filter avoids fake breakouts.

Key design:
1. 12h HMA(21) for macro trend filter (align with mtf_data)
2. 4h HMA(21) for intermediate trend confirmation
3. Choppiness Index(14) for regime detection
4. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
5. Session filter: only trade 8-20 UTC (high liquidity)
6. Volume filter: > 0.8x 20-period average
7. ATR(14) trailing stop at 2.5x for risk management
8. Size: 0.22 discrete levels (smaller for 1h TF)

Target: 40-80 trades/year on 1h, Sharpe > 0.612, trades >= 40 train, >= 5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_crsi_4h12h_hma_session_vol_atr_v1"
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

def calculate_rsi(close, period=3):
    """Relative Strength Index - short period for Connors RSI"""
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

def calculate_rsi_streak(close, period=2):
    """
    Connors RSI Streak Component
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    rsi_streak = np.full(n, np.nan)
    
    for i in range(period, n):
        streak = 0
        if i > 0:
            if close[i] > close[i-1]:
                j = i
                while j > 0 and close[j] > close[j-1]:
                    streak += 1
                    j -= 1
            elif close[i] < close[i-1]:
                j = i
                while j > 0 and close[j] < close[j-1]:
                    streak -= 1
                    j -= 1
        
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            rsi_streak[i] = 100.0 * streak / (period + 1) if streak <= period else 100.0
        else:
            rsi_streak[i] = 100.0 * (period + streak) / (period + 1) if streak >= -period else 0.0
    
    rsi_streak = np.clip(rsi_streak, 0, 100)
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank Component
    Where current price ranks vs last N periods
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + rsi_streak + pr) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index
    CHOP > 61.8 = range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
        
        if atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i-period+1:i+1])
    return vol_sma

def get_hour_from_open_time(open_times):
    """Extract hour from open_time (milliseconds timestamp)"""
    hours = np.zeros(len(open_times), dtype=np.int32)
    for i, ot in enumerate(open_times):
        # Convert ms to seconds, then to datetime
        ts = ot / 1000.0
        hours[i] = int((ts % 86400) / 3600)  # UTC hour
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_times = prices["open_time"].values
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
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    hours = get_hour_from_open_time(open_times)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22
    
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (> 0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === RANGE MARKET: Mean revert with Connors RSI extremes ===
        if is_range and in_session and volume_ok:
            # Long: CRSI < 15 (extreme oversold) + macro not strongly bearish
            if crsi[i] < 15.0 and not macro_bear:
                desired_signal = BASE_SIZE
            # Short: CRSI > 85 (extreme overbought) + macro not strongly bullish
            elif crsi[i] > 85.0 and not macro_bull:
                desired_signal = -BASE_SIZE
            # Moderate mean reversion with trend alignment
            elif crsi[i] < 25.0 and macro_bull and inter_bull:
                desired_signal = BASE_SIZE
            elif crsi[i] > 75.0 and macro_bear and inter_bear:
                desired_signal = -BASE_SIZE
        
        # === TREND MARKET: Follow HTF trend on CRSI pullback ===
        elif is_trend and in_session and volume_ok:
            # Long: Macro bull + 4h bull + CRSI pullback (30-50)
            if macro_bull and inter_bull:
                if 25.0 <= crsi[i] <= 50.0:
                    desired_signal = BASE_SIZE
                # CRSI breaking above 40 with momentum
                elif 35.0 < crsi[i] < 55.0:
                    desired_signal = BASE_SIZE
            
            # Short: Macro bear + 4h bear + CRSI bounce (50-75)
            elif macro_bear and inter_bear:
                if 50.0 <= crsi[i] <= 75.0:
                    desired_signal = -BASE_SIZE
                # CRSI breaking below 60 with momentum
                elif 45.0 < crsi[i] < 65.0:
                    desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE (45 < CHOP < 55): Conservative, need strong confluence ===
        elif in_session and volume_ok:
            # Only enter with strong HTF alignment and CRSI extreme
            if macro_bull and inter_bull and crsi[i] < 20.0:
                desired_signal = BASE_SIZE
            elif macro_bear and inter_bear and crsi[i] > 80.0:
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