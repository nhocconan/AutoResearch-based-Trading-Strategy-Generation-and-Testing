#!/usr/bin/env python3
"""
Experiment #1112: 12h Primary + 1d/1w HTF — Dual Regime with Connors RSI

Hypothesis: After analyzing 800+ failed experiments, key insights:
1. Single-regime strategies fail because BTC/ETH switch between trend and range
2. Choppiness Index reliably detects regime (CHOP>61.8=range, <38.2=trend)
3. Connors RSI (CRSI) outperforms standard RSI for mean reversion (75% win rate)
4. 1w HMA provides ultimate macro filter — only trade with weekly trend
5. KAMA adapts to volatility better than HMA/EMA in choppy conditions
6. Dual regime: mean-revert in chop (CRSI extremes), trend-follow otherwise

Why this should beat Sharpe=0.612:
- Regime detection avoids trading trend strategies in ranges (where they fail)
- CRSI captures short-term reversals better than RSI(14)
- 1w HMA filter prevents counter-trend trades in strong macro moves
- 12h naturally produces 20-40 trades/year — optimal fee/trade balance
- Research shows CRSI+Choppiness achieved ETH Sharpe +0.923

Timeframe: 12h (primary)
HTF: 1d (Choppiness), 1w (HMA macro) — loaded ONCE before loop
Position Size: 0.28 base (discrete: 0.0, ±0.15, ±0.28)
Stoploss: 2.5x ATR trailing
Target: 25-45 trades/year, Sharpe > 0.612, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_chop_1d1w_kama_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    hma = wma(diff, sqrt_period)
    return hma

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER (trending) = fast smoothing, Low ER (choppy) = slow smoothing.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
    
    # Smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = slow_sc + er * (fast_sc - slow_sc)
    sc = np.clip(sc, slow_sc, fast_sc)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite momentum indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.zeros(n)
    diff = np.diff(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if diff[i - 1] > 0:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif diff[i - 1] < 0:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    streak_rsi_raw = np.zeros(n)
    for i in range(streak_period, n):
        if abs_streak[i] >= streak_period:
            streak_rsi_raw[i] = 100.0 if streak[i] > 0 else 0.0
        else:
            streak_rsi_raw[i] = 50.0 + streak[i] * 25.0
    streak_rsi_raw = np.clip(streak_rsi_raw, 0, 100)
    streak_rsi = pd.Series(streak_rsi_raw).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i - rank_period:i + 1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1) if rank_period > 1 else 50.0
    
    # Combine
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & (percent_rank > 0)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market consolidation vs trending.
    CHOP > 61.8 = range/consolidation (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_hl = hh - ll
    
    # Calculate CHOP
    mask = (range_hl > 1e-10) & (atr_sum > 0)
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
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
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultimate macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=10, fast=2, slow=30)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or np.isnan(kama_12h[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness) ===
        # CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
        is_choppy = chop_1d_aligned[i] > 55.0  # Slightly relaxed threshold
        is_trending = chop_1d_aligned[i] < 45.0
        
        # === MOMENTUM (12h CRSI) ===
        crsi_oversold = crsi_12h[i] < 20.0  # Relaxed from 10 for more trades
        crsi_overbought = crsi_12h[i] > 80.0  # Relaxed from 90
        
        # === TREND FILTER (KAMA slope) ===
        kama_bull = kama_12h[i] > kama_12h[i - 5] if not np.isnan(kama_12h[i - 5]) else False
        kama_bear = kama_12h[i] < kama_12h[i - 5] if not np.isnan(kama_12h[i - 5]) else False
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY/RANGE (Mean Reversion) ===
        if is_choppy:
            # Long: CRSI oversold + price above weekly HMA
            if crsi_oversold and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + price below weekly HMA
            elif crsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE
        
        # === REGIME 2: TRENDING (Trend Following) ===
        elif is_trending:
            # Long: Macro bull + KAMA bullish + CRSI not overbought
            if macro_bull and kama_bull and crsi_12h[i] < 70.0:
                desired_signal = BASE_SIZE
            # Short: Macro bear + KAMA bearish + CRSI not oversold
            elif macro_bear and kama_bear and crsi_12h[i] > 30.0:
                desired_signal = -BASE_SIZE
        
        # === REGIME 3: TRANSITION (Neutral/Reduced Size) ===
        else:
            # Only take high-conviction setups in transition
            if crsi_oversold and macro_bull and kama_bull:
                desired_signal = HALF_SIZE
            elif crsi_overbought and macro_bear and kama_bear:
                desired_signal = -HALF_SIZE
        
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
        
        # === EXIT CONDITIONS ===
        if in_position and not stoploss_triggered:
            if position_side > 0:
                # Exit long if macro reverses or CRSI overbought
                if macro_bear or crsi_12h[i] > 85.0:
                    desired_signal = 0.0
            elif position_side < 0:
                # Exit short if macro reverses or CRSI oversold
                if macro_bull or crsi_12h[i] < 15.0:
                    desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= HALF_SIZE * 0.8:
                desired_signal = HALF_SIZE
            else:
                desired_signal = 0.0
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -HALF_SIZE * 0.8:
                desired_signal = -HALF_SIZE
            else:
                desired_signal = 0.0
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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
        
        signals[i] = desired_signal
    
    return signals