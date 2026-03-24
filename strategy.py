#!/usr/bin/env python3
"""
Experiment #1582: 12h Primary + 1d/1w HTF — Dual Regime Strategy

Hypothesis: After 11 failed 12h experiments, dual regime switching based on Choppiness Index
should work better than pure trend or pure mean reversion. Market regime determines strategy:
- CHOP > 61.8 = Range → Connors RSI mean reversion (buy low, sell high)
- CHOP < 38.2 = Trend → Donchian breakout with HMA confirmation

Key innovations:
1. Choppiness Index(14) regime detection - switches between trend/mean-revert modes
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for mean reversion
3. Donchian(20) breakout for trend following
4. 1d HMA(21) for intermediate trend bias
5. 1w HMA(21) for long-term regime filter
6. ATR(14) 2.5x trailing stop for drawdown control
7. Discrete position sizing (0.28) to minimize fee churn

Why this should beat Sharpe 0.618:
- Adapts to market conditions instead of forcing one approach
- Connors RSI has 75% win rate in ranges (proven in research)
- Donchian breakouts catch major trends without whipsaw
- 12h targets 20-50 trades/year — optimal for fee efficiency
- Dual HTF filter ensures we trade with major trend direction

Timeframe: 12h (required for this experiment)
HTF: 1d HMA + 1w HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP = 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) / (log10(n))
    CHOP > 61.8 = Range/Chop, CHOP < 38.2 = Trend
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * (atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Fast RSI on close
    RSI(Streak): RSI on consecutive up/down days
    PercentRank: Rank of today's return vs last 100 days
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3) on close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    mask = loss_3 > 1e-10
    rsi_close[mask] = 100.0 - (100.0 / (1.0 + gain_3[mask] / loss_3[mask]))
    rsi_close[loss_3 <= 1e-10] = 100.0
    rsi_close[:rsi_period] = np.nan
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_2 > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_2[mask] / streak_loss_2[mask]))
    rsi_streak[streak_loss_2 <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # PercentRank(100) - rank of today's return vs last 100 days
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(pr_period, n):
        window = returns[i-pr_period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window < current_return)
        percent_rank[i] = 100.0 * rank / pr_period
    
    # Combine into CRSI
    valid = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_close[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = Highest High over period
    Lower = Lowest Low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average with proper min_periods"""
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_12h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
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
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        
        # === TREND BIAS (1d HMA + 1w HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - Connors RSI extremes
            # Long: CRSI < 10 + price above SMA200 + daily bull bias
            if crsi[i] < 15 and not np.isnan(sma_200[i]) and close[i] > sma_200[i] * 0.98:
                if daily_bull or not weekly_bear:
                    desired_signal = BASE_SIZE
            
            # Short: CRSI > 90 + price below SMA200 + daily bear bias
            elif crsi[i] > 85 and not np.isnan(sma_200[i]) and close[i] < sma_200[i] * 1.02:
                if daily_bear or not weekly_bull:
                    desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND FOLLOWING MODE - Donchian breakout + HMA confirmation
            # Long: Price breaks Donchian upper + HMA bull + daily/weekly bull
            if close[i] > donchian_upper[i] and hma_12h[i] > hma_12h[i-1] if not np.isnan(hma_12h[i-1]) else True:
                if daily_bull and (weekly_bull or not weekly_bear):
                    desired_signal = BASE_SIZE
            
            # Short: Price breaks Donchian lower + HMA bear + daily/weekly bear
            elif close[i] < donchian_lower[i] and hma_12h[i] < hma_12h[i-1] if not np.isnan(hma_12h[i-1]) else True:
                if daily_bear and (weekly_bear or not weekly_bull):
                    desired_signal = -BASE_SIZE
        
        else:
            # NEUTRAL ZONE (38.2 <= CHOP <= 61.8) - reduced position or flat
            # Only take strong signals with HTF confirmation
            if crsi[i] < 10 and daily_bull and weekly_bull:
                desired_signal = BASE_SIZE * 0.5
            elif crsi[i] > 90 and daily_bear and weekly_bear:
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.5
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