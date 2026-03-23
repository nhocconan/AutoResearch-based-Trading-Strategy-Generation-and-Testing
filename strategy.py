#!/usr/bin/env python3
"""
Experiment #471: 4h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + CRSI + KAMA

Hypothesis: Based on research showing regime-adaptive strategies work best in mixed markets.
Key insight from failures: Single-regime strategies fail because 2021-2024 includes both
bull (+219%) and bear (-77%) periods. Need to ADAPT to market conditions.

Innovations:
1. Choppiness Index (14) regime filter: CHOP>50=mean-revert, CHOP<50=trend-follow
2. CRSI for mean reversion: (RSI(3)+RSI_Streak(2)+PercentRank(100))/3 — proven 75% win rate
3. KAMA(14) adaptive trend: adjusts speed based on market efficiency ratio
4. 1d KAMA for HTF bias alignment (load ONCE before loop)
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- Dual regime adapts to bull/bear/range conditions
- CRSI catches reversals in choppy markets (2022 crash bottom, 2025 bear rallies)
- KAMA faster than EMA but less noisy than HMA for crypto
- 4h timeframe targets 20-50 trades/year (optimal fee/trade balance)
- HTF 1d bias prevents counter-trend trades in strong trends

Target: Sharpe > 0.612, 25-50 trades/year, DD < -35%
Timeframe: 4h (proven best for crypto swing trading with HTF alignment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_crsi_kama_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_kama(close, period=14, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Smoothing constant
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    This is a proven mean-reversion indicator with ~75% win rate.
    Long when CRSI < 10, Short when CRSI > 90.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    abs_streak = np.abs(streak)
    for i in range(streak_period, n):
        if abs_streak[i] >= streak_period:
            streak_rsi[i] = 100.0 if streak[i] > 0 else 0.0
        else:
            # Partial streak
            streak_rsi[i] = 50.0 + (streak[i] / streak_period) * 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / (rank_period - 1)
        percent_rank[i] = rank * 100.0
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    choppiness = np.full(n, np.nan)
    
    # ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_percentile_rank(series, window):
    """Calculate rolling percentile rank."""
    n = len(series)
    pr = np.full(n, np.nan)
    
    for i in range(window, n):
        window_vals = series[i-window+1:i+1]
        current = series[i]
        rank = np.sum(window_vals < current) / (window - 1)
        pr[i] = rank * 100.0
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    kama_14 = calculate_kama(close, 14)
    kama_50 = calculate_kama(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    choppiness = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, 21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(kama_14[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            continue
        if np.isnan(kama_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 50.0  # Mean reversion regime
        is_trending = choppiness[i] <= 50.0  # Trend following regime
        
        # === HTF TREND BIAS (1d KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (KAMA crossover) ===
        trend_bullish = kama_14[i] > kama_50[i]
        trend_bearish = kama_14[i] < kama_50[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME: Use CRSI extremes
            # Long: CRSI < 15 (oversold) + HTF not strongly bearish
            if crsi[i] < 15.0 and not (price_below_kama_1d and trend_bearish):
                desired_signal = SIZE_LONG
            
            # Short: CRSI > 85 (overbought) + HTF not strongly bullish
            elif crsi[i] > 85.0 and not (price_above_kama_1d and trend_bullish):
                desired_signal = -SIZE_SHORT
        
        else:
            # TREND FOLLOWING REGIME: Use KAMA + HTF alignment
            # Long: KAMA bullish + HTF bullish + RSI not overbought
            if trend_bullish and price_above_kama_1d and rsi_14[i] < 75.0:
                desired_signal = SIZE_LONG
            
            # Short: KAMA bearish + HTF bearish + RSI not oversold
            elif trend_bearish and price_below_kama_1d and rsi_14[i] > 25.0:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if regime unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if still in valid regime
                if (is_choppy and crsi[i] < 50.0) or (is_trending and trend_bullish and price_above_kama_1d):
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if still in valid regime
                if (is_choppy and crsi[i] > 50.0) or (is_trending and trend_bearish and price_below_kama_1d):
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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