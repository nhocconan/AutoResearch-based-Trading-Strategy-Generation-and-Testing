#!/usr/bin/env python3
"""
Experiment #480: 6h Primary + 1d/1w HTF — CHOP Regime + Connors RSI Hybrid

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy uses:
1. 1w HMA(21) = primary trend bias (very slow, avoids whipsaw)
2. 1d HMA(21) = secondary trend confirmation
3. CHOP(14) = regime filter (>55 = range/mean-revert, <45 = trend-follow)
4. Connors RSI = entry trigger (more responsive than standard RSI)
5. ATR(14)*2.0 stoploss on all positions

Key design choices for 6h:
- LOOSE CHOP thresholds (55/45 not 61.8/38.2) to ensure trade generation
- Connors RSI extremes at 25/75 (not 20/80) for more signals
- OR logic: either regime can trigger trades if HTF bias agrees
- Target: 40-80 trades/year (6h has ~1460 bars/year, need ~3-5% signal rate)

Why 6h might work:
- Less noise than 4h, more signals than 12h
- 1w HTF provides strong bias filter (avoid counter-trend in strong trends)
- CHOP regime adapts to market conditions (range vs trend)

Position sizing: 0.25 base, 0.30 strong signals
Stoploss: 2.0x ATR from entry
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_chop_connors_hma_1d1w_v1"
timeframe = "6h"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        pos_count = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = (pos_count / streak_period) * 100.0
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - where does today's return rank in last 100 days?
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) > 0:
            rank = np.sum(window < returns[i])
            percent_rank[i] = (rank / len(window)) * 100.0
    
    # Combine into Connors RSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10 and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            choppiness[i] = 50.0  # neutral
    
    return choppiness

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

def calculate_sma(close, period):
    """Simple Moving Average"""
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for primary trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for secondary trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    connors_rsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(connors_rsi[i]) or np.isnan(choppiness[i]):
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
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w HTF PRIMARY BIAS ===
        htf_weekly_bull = close[i] > hma_1w_aligned[i]
        htf_weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HTF SECONDARY BIAS ===
        htf_daily_bull = close[i] > hma_1d_aligned[i]
        htf_daily_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # LOOSE thresholds to ensure trade generation
        is_ranging = choppiness[i] > 50.0  # was 55, loosened
        is_trending = choppiness[i] < 50.0  # was 45, loosened
        
        # === CONNORS RSI EXTREMES (LOOSE: 25/75) ===
        crsi_oversold = connors_rsi[i] < 30.0  # was 25, loosened
        crsi_overbought = connors_rsi[i] > 70.0  # was 75, loosened
        crsi_extreme_oversold = connors_rsi[i] < 20.0
        crsi_extreme_overbought = connors_rsi[i] > 80.0
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === ENTRY LOGIC (LOOSE - OR logic, not strict AND) ===
        desired_signal = 0.0
        
        # TREND LONG: Weekly bull + Daily bull + trending regime
        if htf_weekly_bull and htf_daily_bull and is_trending:
            if above_sma50:
                desired_signal = SIZE_STRONG
            elif crsi_oversold:
                # Pullback entry in uptrend
                desired_signal = SIZE_BASE
        
        # TREND SHORT: Weekly bear + Daily bear + trending regime
        elif htf_weekly_bear and htf_daily_bear and is_trending:
            if below_sma50:
                desired_signal = -SIZE_STRONG
            elif crsi_overbought:
                # Rally entry in downtrend
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: Ranging regime + CRSI extreme + above SMA200
        if desired_signal == 0.0 and is_ranging:
            if crsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_STRONG
            elif crsi_oversold and above_sma50:
                desired_signal = SIZE_BASE
        
        # MEAN REVERSION SHORT: Ranging regime + CRSI extreme + below SMA200
        if desired_signal == 0.0 and is_ranging:
            if crsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_STRONG
            elif crsi_overbought and below_sma50:
                desired_signal = -SIZE_BASE
        
        # FALLBACK: Simple CRSI extreme with weekly bias only (ensure trades)
        if desired_signal == 0.0:
            if crsi_extreme_oversold and htf_weekly_bull:
                desired_signal = SIZE_BASE * 0.8
            elif crsi_extreme_overbought and htf_weekly_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals