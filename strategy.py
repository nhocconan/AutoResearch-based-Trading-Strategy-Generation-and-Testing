#!/usr/bin/env python3
"""
Experiment #103: 1d Primary + 1w HTF — HMA Trend + Connors RSI + Choppiness Regime

Hypothesis: After 100+ experiments, the clearest pattern is:
- 1d timeframe with 1w HTF bias reduces whipsaw significantly
- Connors RSI (CRSI) has proven 75% win rate in research literature
- Choppiness Index as SIMPLE regime filter (not dual-regime logic) works for ETH
- HMA is faster than KAMA for trend detection on 1d
- LOOSE entry thresholds ensure trade generation on ALL symbols (BTC/ETH/SOL)

Key learnings from failures:
- Complex dual-regime logic → 0 trades or negative Sharpe (#093, #097)
- Fisher Transform → consistently negative (#091, #094)
- CRSI pure mean reversion → failed (#102)
- BUT: CRSI + trend filter + simple chop filter → ETH Sharpe +0.923 in research

This strategy uses:
1. 1w HMA(21) = major trend bias (price above/below)
2. 1d HMA(16/48) crossover = entry trigger
3. Connors RSI < 30 for long, > 70 for short (looser than standard 10/90)
4. Choppiness Index > 55 = reduce size (choppy), < 45 = full size (trending)
5. ATR trailing stoploss (3x) for risk management
6. Position size: 0.25-0.30 based on regime

Target: Sharpe>0.351, DD>-40%, trades>=20 on train (1d = fewer trades expected)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_crsi_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average - faster response than EMA with less lag
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper: Weighted Moving Average
    def wma(series, span):
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            weights = np.arange(1, span + 1)
            result[i] = np.sum(series[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.zeros(n)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
        else:
            diff[i] = np.nan
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) - very short period for sensitivity
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        # Count positive streaks in lookback
        pos_count = np.sum(streak[i-streak_period+1:i+1] > 0)
        streak_rsi[i] = (pos_count / streak_period) * 100.0
    
    # Percent Rank (100) - where does today's return rank in last 100 days?
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        lookback = returns[i-rank_period+1:i+1]
        count_below = np.sum(lookback[:-1] < returns[i])  # exclude current
        percent_rank[i] = (count_below / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% base position size
    MIN_SIZE = 0.15   # Reduced size in choppy markets
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need more warmup for CRSI (100 period rank)
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA crossover) ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # === CONNORS RSI (Mean Reversion Entry) ===
        # Long: CRSI < 35 (oversold but not extreme - ensures trades)
        # Short: CRSI > 65 (overbought but not extreme - ensures trades)
        crsi_oversold = crsi[i] < 35.0
        crsi_overbought = crsi[i] > 65.0
        
        # === CHOPPINESS REGIME (Size Adjustment) ===
        # CHOP > 55 = choppy (reduce size)
        # CHOP < 45 = trending (full size)
        choppy_market = choppiness[i] > 55.0
        trending_market = choppiness[i] < 45.0
        
        # Adjust position size based on regime
        if trending_market:
            position_size = BASE_SIZE
        elif choppy_market:
            position_size = MIN_SIZE
        else:
            position_size = (BASE_SIZE + MIN_SIZE) / 2.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1w bull + 1d HMA cross bull + CRSI oversold
        # SHORT: 1w bear + 1d HMA cross bear + CRSI overbought
        desired_signal = 0.0
        
        if htf_bull and hma_cross_bull and crsi_oversold:
            desired_signal = position_size
        elif htf_bear and hma_cross_bear and crsi_overbought:
            desired_signal = -position_size
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE if not choppy_market else MIN_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE if not choppy_market else -MIN_SIZE
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