#!/usr/bin/env python3
"""
Experiment #1224: 4h Primary + 12h/1d HTF — Choppiness Regime + Connors RSI + Donchian

Hypothesis: Based on research showing Choppiness Index works well for ETH (Sharpe +0.923) and
Connors RSI has 75% win rate for mean reversion. Combine with Donchian breakout for trend phases.
Key insight: Use CHOP to SWITCH between regimes rather than as a filter. CHOP>61.8 = range (use CRSI),
CHOP<38.2 = trend (use Donchian). This avoids the whipsaw problem from #1212/#1221.

Why this should work:
1. 4h TF proven to work best (current best Sharpe=0.612 is 4h)
2. Choppiness regime switch is simpler than 3-regime system (#1212 failed with complexity)
3. Connors RSI faster than standard RSI for mean reversion entries
4. Donchian(20) breakout catches trending moves without lag
5. 12h/1d HMA for macro trend filter (only trade with HTF trend)
6. Conservative size (0.25) + 2.5x ATR stop for drawdown control

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_donchian_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
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
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Max High - Min Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_rsi = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_rsi = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = loss_rsi > 1e-10
    rs[mask] = gain_rsi[mask] / loss_rsi[mask]
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close[:rsi_period] = np.nan
    
    # RSI(streak, 2)
    streak = np.zeros(n)
    streak[0] = 0
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    gain_streak = np.where(streak > 0, streak, 0)
    loss_streak = np.where(streak < 0, -streak, 0)
    
    gain_streak_smooth = pd.Series(gain_streak).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    loss_streak_smooth = pd.Series(loss_streak).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = np.zeros(n)
    mask_streak = loss_streak_smooth > 1e-10
    rs_streak[mask_streak] = gain_streak_smooth[mask_streak] / loss_streak_smooth[mask_streak]
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak[:streak_period] = np.nan
    
    # PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 12h HMA for intermediate trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === MACRO TREND FILTER (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (12h HMA) ===
        inter_bull = close[i] > hma_12h_aligned[i]
        inter_bear = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 55.0  # Range-bound (mean reversion)
        is_trending = choppiness[i] < 45.0  # Trending (breakout)
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # MEAN REVERSION MODE (choppy market)
        if is_choppy:
            # Long: CRSI oversold + price above macro HMA (bullish bias)
            if crsi[i] < 15.0 and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + price below macro HMA (bearish bias)
            elif crsi[i] > 85.0 and macro_bear:
                desired_signal = -BASE_SIZE
        
        # TREND FOLLOWING MODE (trending market)
        elif is_trending:
            # Long: Donchian breakout + aligned trends
            if close[i] > donchian_upper[i - 1] and macro_bull and inter_bull:
                desired_signal = BASE_SIZE
            # Short: Donchian breakdown + aligned trends
            elif close[i] < donchian_lower[i - 1] and macro_bear and inter_bear:
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
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals