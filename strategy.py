#!/usr/bin/env python3
"""
Experiment #057: 1d Primary + 1w HTF — Choppiness Index Regime + Connors RSI

Hypothesis: After 56 experiments, the key insight is REGIME DETECTION matters most.
Research shows Choppiness Index + Connors RSI achieved ETH Sharpe +0.923.

Why this should work:
1. 1d timeframe = 10-30 trades/year (fee-efficient, matches research)
2. 1w Choppiness Index identifies regime: CHOP>61.8=range(mean revert), CHOP<38.2=trend
3. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI for mean reversion
   - Proven 75% win rate in research
4. 1w HMA(21) for trend bias when in trending regime
5. LOOSE CRSI thresholds (15/85 not 10/90) to ensure trades generate
6. Discrete sizing 0.30, ATR 2.5x trailing stop

Entry Logic:
- Range regime (1w CHOP > 55): Long CRSI<20, Short CRSI>80
- Trend regime (1w CHOP < 45): Long if price>1w HMA + CRSI<40, Short if price<1w HMA + CRSI>60
- Size: 0.30 discrete
- Stop: 2.5x ATR trailing

Target: Sharpe>0.37 (beat current best), trades>10/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies ranging vs trending markets
    Formula: 100 * (SUM(ATR(1), n) / (Highest High - Lowest Low)) * 100 / log10(n)
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
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
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        if hh - ll > 1e-10:
            chop[i] = 100.0 * (tr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak - consecutive up/down days
    3. PercentRank(100) - where current price ranks vs last 100 days
    
    Long: CRSI < 10-20 | Short: CRSI > 80-90
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3) on close
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
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak (treating positive as gains, negative as losses)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        total_loss = avg_streak_loss[i]
        if total_loss < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / total_loss
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i - pr_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / pr_period
    
    # Combine components
    crsi = np.full(n, np.nan)
    for i in range(pr_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smooth trend indicator"""
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
    
    # Calculate and align 1w indicators for regime detection
    chop_1w_raw = calculate_choppiness(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after CRSI warmup (100) + buffer
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop_1w_aligned[i]) or np.isnan(hma_1w_aligned[i]):
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
        
        # === REGIME DETECTION (1w Choppiness) ===
        chop_value = chop_1w_aligned[i]
        
        # Range regime: CHOP > 55 (mean reversion works best)
        is_range_regime = chop_value > 55.0
        
        # Trend regime: CHOP < 45 (trend following works best)
        is_trend_regime = chop_value < 45.0
        
        # Neutral regime: 45 <= CHOP <= 55 (reduce position or stay flat)
        is_neutral_regime = not is_range_regime and not is_trend_regime
        
        # === TREND BIAS (1w HMA) ===
        price_above_hma = close[i] > hma_1w_aligned[i]
        price_below_hma = close[i] < hma_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_value = crsi[i]
        
        # Range regime: pure mean reversion at extremes
        crsi_oversold = crsi_value < 20.0  # Long signal in range
        crsi_overbought = crsi_value > 80.0  # Short signal in range
        
        # Trend regime: mean reversion WITH trend bias
        crsi_pullback_long = crsi_value < 40.0  # Long pullback in uptrend
        crsi_pullback_short = crsi_value > 60.0  # Short pullback in downtrend
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_range_regime:
            # Mean reversion in range
            if crsi_oversold:
                desired_signal = SIZE
            elif crsi_overbought:
                desired_signal = -SIZE
        
        elif is_trend_regime:
            # Trend following with pullback entries
            if price_above_hma and crsi_pullback_long:
                desired_signal = SIZE
            elif price_below_hma and crsi_pullback_short:
                desired_signal = -SIZE
        
        # Neutral regime: reduce existing positions or stay flat
        # (desired_signal stays 0.0)
        
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