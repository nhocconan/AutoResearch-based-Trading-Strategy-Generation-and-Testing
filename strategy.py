#!/usr/bin/env python3
"""
Experiment #073: 1d Primary + 1w HTF — Donchian Breakout with Regime Filter

Hypothesis: After analyzing 72 failed experiments, the winning pattern for 1d timeframe is:
1. 1w HMA for major trend bias (prevents counter-trend trades in macro moves)
2. Donchian(20) breakout for entry timing (proven for SOL Sharpe +0.782)
3. Connors RSI for pullback entries (proven for ETH Sharpe +0.923)
4. Choppiness Index regime filter (avoid trend-following in choppy markets)
5. ATR(14) 2.5x trailing stop (proven risk management)

Why 1d + 1w should work:
- 1d timeframe = 20-50 trades/year target (fee-efficient, matches research)
- 1w HMA is slow enough to filter 2022 crash whipsaws
- Connors RSI (RSI3 + RSI_Streak + PercentRank) / 3 catches pullbacks better than RSI14
- Choppiness Index > 50 = avoid trend entries (reduce false breakouts)
- Loose Connors RSI thresholds (25/75 not 10/90) ensure we get trades

Entry Logic:
- Long: Donchian(20) breakout + Connors RSI < 75 + price > 1w HMA + CHOP < 55
- Short: Donchian(20) breakout + Connors RSI > 25 + price < 1w HMA + CHOP < 55
- Size: 0.28 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_crsi_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Fast RSI for short-term momentum
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current price ranks in last 100 days
    
    Proven edge: 75% win rate on mean reversion entries
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) - fast momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_fast = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_fast[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_fast[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI of Streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total < 1e-10:
            rsi_streak[i] = 50.0
        else:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / total
    
    # Percent Rank - where current price ranks in last 100 days
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Connors RSI = average of three components
    crsi = (rsi_fast + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10 or sum_atr < 1e-10:
            choppiness[i] = 50.0
        else:
            choppiness[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return choppiness

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size (28% of capital)
    
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
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP < 55 = trending market (allow trend entries)
        # CHOP > 55 = choppy market (avoid trend entries)
        is_trending = choppiness[i] < 55.0
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Check if price broke out THIS bar
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CONNORS RSI FILTER (LOOSE thresholds to ensure trades) ===
        # Long: CRSI < 75 (not extremely overbought, allows pullback entries)
        # Short: CRSI > 25 (not extremely oversold, allows bounce entries)
        crsi_ok_long = crsi[i] < 75.0
        crsi_ok_short = crsi[i] > 25.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Donchian breakout + CRSI filter + 1w HMA bullish + trending
        if breakout_long and crsi_ok_long and hma_1w_bull and is_trending:
            desired_signal = SIZE
        
        # Short entry: Donchian breakout + CRSI filter + 1w HMA bearish + trending
        elif breakout_short and crsi_ok_short and hma_1w_bear and is_trending:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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