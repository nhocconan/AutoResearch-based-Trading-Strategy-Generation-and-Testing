#!/usr/bin/env python3
"""
Experiment #086: 1d Primary + 1w HTF — Connors RSI + Choppiness + HMA + KAMA Dual Regime

Hypothesis: After 85 failed experiments, the clearest pattern is:
- 1d timeframe with 1w HTF works best (current best: Sharpe=0.167)
- Connors RSI (CRSI) captures mean-reversion extremes better than standard RSI
- Choppiness Index regime switching prevents trend-following in ranges
- KAMA adapts to volatility better than HMA alone
- Dual regime: trend-follow when CHOP<45+ADX>25, mean-revert when CHOP>55
- LOOSE entry filters to ensure >=30 trades on train, >=3 on test

Key design choices:
- Timeframe: 1d (20-50 trades/year target)
- HTF: 1w HMA(21) for major trend bias
- Entry: CRSI extremes (15/85) + regime filter + HMA/KAMA confluence
- Regime: CHOP>55 = mean revert, CHOP<45+ADX>25 = trend follow
- Position size: 0.30 (30% of capital, conservative for 1d swings)
- Stoploss: 2.5x ATR trailing
- Relaxed CRSI thresholds (15/85 vs 10/90) to ensure trade frequency

Target: Sharpe>0.167, DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_kama_1w_regime_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    
    CRSI < 10-15 = extremely oversold (long)
    CRSI > 85-90 = extremely overbought (short)
    """
    n = len(close)
    if n < pr_period + rsi_period:
        return np.full(n, np.nan)
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    rsi_close[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI(Streak, 2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_rsi_input = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi_input[i] = 100.0
        elif streak[i] < 0:
            streak_rsi_input[i] = 0.0
        else:
            streak_rsi_input[i] = 50.0
    
    # Smooth streak RSI
    streak_rsi = pd.Series(streak_rsi_input).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Percent Rank (100)
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(pr_period, n):
        lookback = close[i-pr_period+1:i+1]
        count_below = np.sum(lookback < close[i])
        pr[i] = 100.0 * count_below / pr_period
    
    # Combine
    crsi = np.zeros(n)
    crsi[:] = np.nan
    valid = ~np.isnan(rsi_close) & ~np.isnan(streak_rsi) & ~np.isnan(pr)
    crsi[valid] = (rsi_close[valid] + streak_rsi[valid] + pr[valid]) / 3.0
    
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use 55/45 as thresholds for regime switching
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < er_period + slow_period:
        return kama
    
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i-er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(prices, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(prices)
    if n < period * 3:
        return np.full(n, np.nan)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    sm_plus_dm = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    sm_minus_dm = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    sm_plus_di = 100.0 * sm_plus_dm / (atr + 1e-10)
    sm_minus_di = 100.0 * sm_minus_dm / (atr + 1e-10)
    
    dx = 100.0 * np.abs(sm_plus_di - sm_minus_di) / (sm_plus_di + sm_minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx = calculate_adx(prices, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 1d)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # Choppiness: >55 = choppy/range, <45 = trending
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # ADX: >25 = strong trend, <20 = weak/no trend
        strong_trend = adx[i] > 25.0
        weak_trend = adx[i] < 20.0
        
        # === TREND FILTER (HMA) ===
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === CONNORS RSI SIGNALS (LOOSE for trade frequency) ===
        crsi_oversold = crsi[i] < 20.0  # Relaxed from 15
        crsi_overbought = crsi[i] > 80.0  # Relaxed from 85
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending or strong_trend:
            # TREND REGIME: Follow trend with pullback entries
            if htf_bull and hma_bull and kama_bull and rsi[i] > 35.0 and rsi[i] < 65.0:
                desired_signal = SIZE
            elif htf_bear and hma_bear and kama_bear and rsi[i] < 65.0 and rsi[i] > 35.0:
                desired_signal = -SIZE
            # CRSI extreme in trend direction
            elif htf_bull and hma_bull and crsi_oversold:
                desired_signal = SIZE
            elif htf_bear and hma_bear and crsi_overbought:
                desired_signal = -SIZE
        elif is_choppy or weak_trend:
            # CHOPPY REGIME: Mean reversion at extremes
            if crsi_oversold and not htf_bear:
                desired_signal = SIZE
            elif crsi_overbought and not htf_bull:
                desired_signal = -SIZE
            # RSI extreme mean reversion
            elif rsi_oversold and hma_bull:
                desired_signal = SIZE * 0.7
            elif rsi_overbought and hma_bear:
                desired_signal = -SIZE * 0.7
        else:
            # NEUTRAL REGIME: Wait for CRSI extremes
            if crsi_oversold:
                desired_signal = SIZE * 0.7
            elif crsi_overbought:
                desired_signal = -SIZE * 0.7
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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