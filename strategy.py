#!/usr/bin/env python3
"""
Experiment #1642: 12h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: 12h timeframe proven to work better than 4h (see #1632, #1636 kept). 
Previous failures due to: (1) too many confluence filters = 0 trades, (2) pure trend 
following fails in bear/range markets (2022 crash, 2025 bear).

This strategy uses DUAL REGIME approach:
1. CHOPPINESS INDEX (14) detects regime: CHOP>61.8=range, CHOP<38.2=trend
2. RANGE REGIME: Connors RSI mean reversion (CRSI<10 long, CRSI>90 short)
3. TREND REGIME: Donchian(20) breakout + 1d HMA bias + KAMA confirmation
4. 1w HMA for meta-trend filter (only trade with weekly direction)
5. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- 12h targets 20-50 trades/year = optimal fee/trade balance
- Dual regime adapts to market conditions (trend vs range)
- Connors RSI has 75% win rate in range markets
- Donchian breakout catches major trends
- Weekly HMA prevents counter-trend trades in strong regimes
- Simplified entry logic = MORE TRADES (avoid 0 trade failure)

Timeframe: 12h (required)
HTF: 1d HMA + 1w HMA via mtf_data.get_htf_data() — called ONCE before loop
Target: Sharpe > 0.618, trades > 30/symbol train, > 5/symbol test, DD > -40%
Size: 0.25 (discrete: 0.0, ±0.25)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < slow_period + er_period:
        return np.full(n, np.nan)
    
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        if price_change < 1e-10:
            er[i] = 0.0
        else:
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            er[i] = price_change / volatility if volatility > 1e-10 else 0.0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Mean reversion indicator - extremes indicate reversal
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent Rank: where does current return rank vs last 100 bars
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / close[:-1]
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i-1]
        pct_rank[i] = np.sum(window < current) / len(window) * 100.0
    
    # Combine
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + pct_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures if market is trending or ranging
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        if highest_high - lowest_low < 1e-10:
            chop[i] = 100.0
        else:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel: upper = highest high, lower = lowest low"""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market (mean revert)
        is_trending = chop[i] < 45.0  # Trend market (breakout)
        
        # === HTF TREND BIAS ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # RANGE REGIME: Connors RSI Mean Reversion
        if is_choppy:
            # Long: CRSI < 15 (oversold) + price above weekly HMA
            if crsi[i] < 15.0 and weekly_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI > 85 (overbought) + price below weekly HMA
            elif crsi[i] > 85.0 and weekly_bear:
                desired_signal = -BASE_SIZE
        
        # TREND REGIME: Donchian Breakout
        elif is_trending:
            # Long breakout: price breaks Donchian upper + daily/weekly bull + above KAMA
            if close[i] > donchian_upper[i-1] and daily_bull and weekly_bull and close[i] > kama[i]:
                desired_signal = BASE_SIZE
            # Short breakout: price breaks Donchian lower + daily/weekly bear + below KAMA
            elif close[i] < donchian_lower[i-1] and daily_bear and weekly_bear and close[i] < kama[i]:
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
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