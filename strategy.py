#!/usr/bin/env python3
"""
Experiment #027: 1d Primary + 1w HTF — Regime-Adaptive Dual Strategy

Hypothesis: After 23 failed experiments, the key issue is single-regime strategies.
Markets alternate between trending and choppy. A strategy that adapts to regime
should work better than pure trend-following or pure mean-reversion.

Key innovations:
1. CHOPPINESS INDEX (14) regime detection: CHOP>61.8=chop, CHOP<38.2=trend
2. DUAL MODE: Mean-revert in chop (Connors RSI), trend-follow in trend (HMA)
3. 1w HMA for ultra-HTF bias - ensures we trade with secular trend
4. LOOSE thresholds: CRSI<20/>80 (vs standard 10/90), HMA cross with any RSI
5. Size: 0.28 discrete - balances return vs drawdown

Entry Logic:
- CHOPPY (CHOP>61.8): Long if CRSI<20 + price>SMA200, Short if CRSI>80 + price<SMA200
- TRENDING (CHOP<38.2): Long if price>HMA21 + 1w HMA bullish, Short if opposite
- TRANSITION (38.2-61.8): Flat or reduce size to 0.15

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.25, trades>20/symbol train, >3/symbol test, DD>-35%
Timeframe: 1d (target 20-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_chop_dual_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10 or sum_atr < 1e-10:
            choppiness[i] = 100.0
        else:
            choppiness[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    if n < rank_period + rsi_period + streak_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # Component 1: RSI(3) of close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: Percent Rank of daily returns over 100 days
    percent_rank = np.full(n, np.nan)
    daily_returns = np.diff(close) / close[:-1]
    daily_returns = np.concatenate([[0.0], daily_returns])
    
    for i in range(rank_period, n):
        window = daily_returns[i-rank_period+1:i+1]
        current = daily_returns[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine all three components
    for i in range(max(period, rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smooth and responsive"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(data, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(data), np.nan)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    wma_diff = 2.0 * wma_half - wma_full
    
    hma = wma(wma_diff, sqrt_period)
    return hma

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

def calculate_sma(close, period=200):
    """Simple Moving Average - for trend filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultra-HTF trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_NORMAL = 0.28  # Normal position size
    SIZE_REDUCED = 0.15  # Reduced size in transition regime
    
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
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_21[i]) or np.isnan(sma_200[i]):
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
        
        # === REGIME DETECTION ===
        chop_value = choppiness[i]
        
        # CHOPPY regime: CHOP > 61.8 (mean reversion mode)
        is_choppy = chop_value > 61.8
        
        # TRENDING regime: CHOP < 38.2 (trend following mode)
        is_trending = chop_value < 38.2
        
        # TRANSITION regime: 38.2 <= CHOP <= 61.8 (reduced size or flat)
        is_transition = not is_choppy and not is_trending
        
        # === HTF TREND BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        current_size = SIZE_NORMAL if not is_transition else SIZE_REDUCED
        
        if is_choppy:
            # MEAN REVERSION MODE (Connors RSI)
            # Long: CRSI < 20 (oversold) + price above SMA200 (long-term uptrend)
            if crsi[i] < 20.0 and not np.isnan(sma_200[i]) and close[i] > sma_200[i]:
                desired_signal = current_size
            
            # Short: CRSI > 80 (overbought) + price below SMA200 (long-term downtrend)
            elif crsi[i] > 80.0 and not np.isnan(sma_200[i]) and close[i] < sma_200[i]:
                desired_signal = -current_size
        
        elif is_trending:
            # TREND FOLLOWING MODE (HMA + 1w bias)
            hma_21_bull = close[i] > hma_21[i]
            hma_21_bear = close[i] < hma_21[i]
            
            # Long: 1d HMA bullish + 1w HMA bullish alignment
            if hma_21_bull and hma_1w_bull:
                desired_signal = current_size
            
            # Short: 1d HMA bearish + 1w HMA bearish alignment
            elif hma_21_bear and hma_1w_bear:
                desired_signal = -current_size
        
        # TRANSITION regime: stay flat or maintain existing position
        # (don't open new positions, but don't force exit either)
        
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
        if desired_signal >= SIZE_NORMAL * 0.85:
            final_signal = SIZE_NORMAL
        elif desired_signal >= SIZE_REDUCED * 0.85:
            final_signal = SIZE_REDUCED
        elif desired_signal <= -SIZE_NORMAL * 0.85:
            final_signal = -SIZE_NORMAL
        elif desired_signal <= -SIZE_REDUCED * 0.85:
            final_signal = -SIZE_REDUCED
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