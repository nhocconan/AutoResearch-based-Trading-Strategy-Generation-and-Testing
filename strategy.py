#!/usr/bin/env python3
"""
Experiment #1332: 12h Primary + 1d/1w HTF — Regime Adaptive (Chop/Trend) + CRSI/Donchian

Hypothesis: 12h timeframe balances trade frequency (20-50/year) with signal quality.
Key insight from failures: BTC/ETH need regime detection. Simple trend fails in 2022 crash.
This strategy switches logic based on Choppiness Index:
- CHOP > 61.8: Range market → Connors RSI mean reversion (75% win rate)
- CHOP < 38.2: Trend market → Donchian breakout with HMA filter
- Between: No trades (avoid whipsaw)

Multi-timeframe design:
1. 1d HMA(21) for macro bias (long only above, short only below)
2. 1w HMA(21) for ultra-macro filter (avoid counter-trend trades)
3. 12h Choppiness(14) for regime detection
4. 12h Connors RSI for mean reversion entries
5. 12h Donchian(20) for breakout entries
6. ATR(14) trailing stop 2.5x for risk management
7. Size: 0.25 discrete (conservative for 12h)

Target: 25-45 trades/year, Sharpe > 0.612, trades >= 30 train, >= 5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_crsi_donchian_1d1w_hma_atr_v1"
timeframe = "12h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - identifies ranging vs trending markets
    CHOP > 61.8: Range/consolidation (mean revert)
    CHOP < 38.2: Trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest High and Lowest Low over period
    hh_ll = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = np.nanmax(high[i-period+1:i+1])
        ll = np.nanmin(low[i-period+1:i+1])
        hh_ll[i] = hh - ll
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(atr_sum[i]) and not np.isnan(hh_ll[i]) and hh_ll[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / hh_ll[i]) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    """
    n = len(close)
    if n < rank_period + rsi_period + streak_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        # Component 1: RSI(3) of close
        delta = np.diff(close[max(0, i-rsi_period-5):i+1])
        if len(delta) < rsi_period:
            continue
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains[-rsi_period:]) if len(gains) >= rsi_period else 0
        avg_loss = np.mean(losses[-rsi_period:]) if len(losses) >= rsi_period else 1e-10
        rsi_close = 100.0 - (100.0 / (1.0 + avg_gain / (avg_loss + 1e-10)))
        
        # Component 2: RSI of streak (consecutive up/down days)
        streak = 0
        for j in range(i, max(0, i-20), -1):
            if j == i:
                continue
            if close[j] > close[j-1]:
                streak += 1
            elif close[j] < close[j-1]:
                streak -= 1
            else:
                break
        # Calculate RSI of streak values over last streak_period bars
        streak_vals = []
        for k in range(max(0, i-streak_period-5), i+1):
            s = 0
            for m in range(k, max(0, k-20), -1):
                if m == k:
                    continue
                if close[m] > close[m-1]:
                    s += 1
                elif close[m] < close[m-1]:
                    s -= 1
                else:
                    break
            streak_vals.append(s)
        
        if len(streak_vals) >= streak_period + 1:
            streak_delta = np.diff(streak_vals[-streak_period-1:])
            streak_gains = np.where(streak_delta > 0, streak_delta, 0)
            streak_losses = np.where(streak_delta < 0, -streak_delta, 0)
            streak_avg_gain = np.mean(streak_gains) if len(streak_gains) > 0 else 0
            streak_avg_loss = np.mean(streak_losses) if len(streak_losses) > 0 else 1e-10
            rsi_streak = 100.0 - (100.0 / (1.0 + streak_avg_gain / (streak_avg_loss + 1e-10)))
        else:
            rsi_streak = 50.0
        
        # Component 3: PercentRank of close over rank_period
        window = close[max(0, i-rank_period+1):i+1]
        if len(window) >= rank_period:
            rank = np.sum(window[:-1] < close[i]) / (len(window) - 1) * 100.0
        else:
            rank = 50.0
        
        crsi[i] = (rsi_close + rsi_streak + rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout detection"""
    n = len(close) if 'close' in dir() else len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-macro filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === ULTRA-MACRO FILTER (1w HMA) ===
        # Only take longs if above weekly HMA, shorts if below
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Slightly lower threshold for more trades
        is_trend = chop[i] < 45.0  # Slightly higher threshold for more trades
        
        desired_signal = 0.0
        
        # === RANGE REGIME: Connors RSI Mean Reversion ===
        if is_range:
            # Long: CRSI oversold + macro bias support
            if crsi[i] < 20.0 and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + macro bias resistance
            elif crsi[i] > 80.0 and macro_bear:
                desired_signal = -BASE_SIZE
            # Extreme mean reversion (ignore macro at extremes)
            elif crsi[i] < 10.0:
                desired_signal = BASE_SIZE
            elif crsi[i] > 90.0:
                desired_signal = -BASE_SIZE
        
        # === TREND REGIME: Donchian Breakout ===
        elif is_trend:
            # Long breakout: price breaks Donchian upper + macro bull + weekly bull
            if close[i] >= donchian_upper[i] and macro_bull and weekly_bull:
                desired_signal = BASE_SIZE
            # Short breakout: price breaks Donchian lower + macro bear + weekly bear
            elif close[i] <= donchian_lower[i] and macro_bear and weekly_bear:
                desired_signal = -BASE_SIZE
            # Pullback entry in trend: price near Donchian lower in uptrend
            elif macro_bull and weekly_bull:
                dist_to_lower = (close[i] - donchian_lower[i]) / (donchian_upper[i] - donchian_lower[i] + 1e-10)
                if dist_to_lower < 0.15 and crsi[i] < 40.0:
                    desired_signal = BASE_SIZE
            # Pullback entry in downtrend
            elif macro_bear and weekly_bear:
                dist_to_upper = (donchian_upper[i] - close[i]) / (donchian_upper[i] - donchian_lower[i] + 1e-10)
                if dist_to_upper < 0.15 and crsi[i] > 60.0:
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