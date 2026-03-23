#!/usr/bin/env python3
"""
Experiment #1416: 12h Primary + 1d HTF — Dual Regime (Chop/Trend) + Connors RSI + Donchian

Hypothesis: 12h timeframe with regime-switching logic will outperform pure trend-following.
Research shows Choppiness Index + Connors RSI achieved Sharpe +0.923 on ETH, and
Donchian + HMA + RSI + ATR achieved Sharpe +0.782 on SOL. Combining these into a
dual-regime system should:
1. Mean-revert in choppy markets (CHOP > 61.8) using Connors RSI extremes
2. Trend-follow in trending markets (CHOP < 38.2) using Donchian breakouts
3. Use 1d HMA(21) as macro trend filter for direction bias
4. ATR(14) trailing stop 2.5x for risk management
5. Position size 0.28 for conservative 12h volatility

Target: 20-50 trades/year, Sharpe > 0.618 (beat 1d baseline), trades >= 30 train, >= 5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d_atr_v1"
timeframe = "12h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - detects ranging vs trending markets
    CHOP > 61.8 = choppy/ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        range_val = highest - lowest
        if range_val > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / range_val) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_rsi = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_rsi = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    mask = loss_rsi > 1e-10
    rsi_close[mask] = 100.0 - (100.0 / (1.0 + gain_rsi[mask] / loss_rsi[mask]))
    rsi_close[loss_rsi <= 1e-10] = 100.0
    rsi_close[:rsi_period] = np.nan
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_positive = np.where(streak > 0, streak, 0.0)
    streak_negative = np.where(streak < 0, -streak, 0.0)
    
    gain_streak = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    loss_streak = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = loss_streak > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + gain_streak[mask] / loss_streak[mask]))
    rsi_streak[loss_streak <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # Component 3: Percent Rank of daily returns over 100 periods
    percent_rank = np.full(n, np.nan)
    daily_returns = np.diff(close, prepend=close[0]) / (close[0] + 1e-10)
    
    for i in range(rank_period, n):
        window = daily_returns[i-rank_period+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window < daily_returns[i])
            percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    donchian_55_upper, donchian_55_lower = calculate_donchian(high, low, period=55)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
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
        if np.isnan(donchian_20_upper[i]) or np.isnan(donchian_55_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
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
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market - mean revert
        is_trending = chop[i] < 38.2  # Trend market - trend follow
        # Neutral zone: 38.2 <= CHOP <= 61.8 - use lighter position or wait
        
        # === MACRO TREND (1d HMA) - direction bias ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === DESIRED SIGNAL - DUAL REGIME LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY MARKET (Mean Reversion with Connors RSI)
        if is_choppy:
            # Long: CRSI < 10 (extreme oversold) + price above 1d HMA (bullish bias)
            if crsi[i] < 10.0 and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI > 90 (extreme overbought) + price below 1d HMA (bearish bias)
            elif crsi[i] > 90.0 and macro_bear:
                desired_signal = -BASE_SIZE
            # Weaker signals in neutral zone (CRSI extremes without HMA confirmation)
            elif crsi[i] < 5.0:
                desired_signal = BASE_SIZE * 0.5
            elif crsi[i] > 95.0:
                desired_signal = -BASE_SIZE * 0.5
        
        # REGIME 2: TRENDING MARKET (Breakout with Donchian)
        elif is_trending:
            # Long: Donchian-20 breakout + 1d HMA bull
            if close[i] > donchian_20_upper[i-1] and macro_bull:
                desired_signal = BASE_SIZE
            # Short: Donchian-20 breakdown + 1d HMA bear
            elif close[i] < donchian_20_lower[i-1] and macro_bear:
                desired_signal = -BASE_SIZE
            # Donchian-55 for stronger trends
            elif close[i] > donchian_55_upper[i-1] and macro_bull:
                desired_signal = BASE_SIZE
            elif close[i] < donchian_55_lower[i-1] and macro_bear:
                desired_signal = -BASE_SIZE
        
        # REGIME 3: NEUTRAL (lighter positions on extremes)
        else:
            # Only take strongest CRSI signals in neutral regime
            if crsi[i] < 5.0 and macro_bull:
                desired_signal = BASE_SIZE * 0.5
            elif crsi[i] > 95.0 and macro_bear:
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