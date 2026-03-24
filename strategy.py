#!/usr/bin/env python3
"""
Experiment #1595: 1h Primary + 4h/1d HTF — CRSI Mean Reversion with Trend Filter

Hypothesis: After 12 consecutive failures with Sharpe=0.000 (ZERO TRADES), the problem is
entry conditions that are TOO STRICT. This strategy simplifies to ensure trades fire.

Key innovations:
1. 4h HMA(21) for trend bias ONLY (not entry trigger) — simple, proven
2. 1h Connors RSI (CRSI) for mean reversion entries — 75% win rate in literature
3. CHOP(14) regime filter — but with WIDE thresholds to allow trades
4. Relaxed CRSI thresholds (15/85 not 10/90) to generate sufficient signals
5. ATR(14) 2.5x trailing stop for drawdown control
6. Position size 0.25 (conservative for 1h TF with 30-60 trades/year target)

CRSI Formula (Connors RSI):
- RSI(3): 3-period RSI for short-term momentum
- RSI_Streak(2): RSI of up/down streak length
- PercentRank(100): percentile of price change over 100 periods
- CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

Entry Logic (LOOSE to ensure trades):
- Long: price > 4h_HMA + CRSI < 20 (oversold in uptrend)
- Short: price < 4h_HMA + CRSI > 80 (overbought in downtrend)
- CHOP > 45 filter (range/trend neutral — very permissive)

Why this should beat Sharpe=0.000 failures:
- CRSI < 20 / > 80 happens ~10% of time = ~876 signals/year on 1h
- With 4h trend filter (~50% filter) = ~438 potential entries/year
- With CHOP filter (~70% pass) = ~306 entries/year across 3 symbols = ~100/symbol/year
- Target: 30-60 trades/symbol/year after signal discretization and stoploss

Timeframe: 1h (required for this experiment)
HTF: 4h HMA for trend bias (use mtf_data helper — call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_chop_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — proven mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down day streak length
    PercentRank: percentile rank of 1-period price change over lookback
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi3 = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi3[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi3[loss_smooth <= 1e-10] = 100.0
    rsi3[:rsi_period] = np.nan
    
    # Streak RSI
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Calculate RSI of absolute streak values
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0.0)
    streak_loss = np.where(streak < 0, streak_abs, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period, n):
        if streak_loss_smooth[i] > 1e-10:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[i] / streak_loss_smooth[i]))
        else:
            streak_rsi[i] = 100.0
    
    # Percent Rank (100)
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns[:-1] <= current_return)
            pct_rank[i] = 100.0 * rank / max(len(returns) - 1, 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is chopping (range-bound) or trending
    CHOP > 61.8 = range-bound, CHOP < 38.2 = trending
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        if hh[i] - ll[i] > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h TF
    
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
        if np.isnan(hma_4h_aligned[i]):
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
        
        # === TREND BIAS (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === CHOP REGIME (permissive threshold) ===
        chop_range = chop[i] > 45.0  # Wide threshold to allow trades
        
        # === CRSI MEAN REVERSION (relaxed thresholds) ===
        crsi_oversold = crsi[i] < 20.0  # Relaxed from 15
        crsi_overbought = crsi[i] > 80.0  # Relaxed from 85
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG: uptrend + oversold CRSI + range/trend neutral
        if trend_bull and crsi_oversold and chop_range:
            desired_signal = BASE_SIZE
        
        # SHORT: downtrend + overbought CRSI + range/trend neutral
        elif trend_bear and crsi_overbought and chop_range:
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