#!/usr/bin/env python3
"""
Experiment #482: 12h Primary + 1d/1w HTF — HMA Trend + Connors RSI + Choppiness Regime

Hypothesis: Building on #477's success (Sharpe=0.092) but simplifying entry logic and using
Connors RSI (CRSI) instead of standard RSI. Research shows CRSI has 75% win rate for mean
reversion entries. Key changes from #477:
1. HMA(21) instead of KAMA - faster trend response, less lag
2. Connors RSI (RSI3 + StreakRSI2 + PercentRank100) / 3 - proven mean reversion signal
3. Simpler scoring: HTF bias + primary trend + CRSI extreme = entry (no complex thresholds)
4. Dual HTF: 1d for intermediate trend, 1w for major bias (both must align)
5. Choppiness Index for regime: CHOP>61.8=range (use CRSI extremes), CHOP<38.2=trend (use pullbacks)
6. Relaxed CRSI thresholds: <15 for long, >85 for short (vs original <10/>90 which was too strict)
7. Position sizing: 0.30 long, 0.25 short (asymmetric, discrete)
8. ATR(14) trailing stop at 2.5x

Why this should work: #477 proved 1d/12h with KAMA+Chop+RSI can generate positive returns.
HMA is more responsive than KAMA for trend detection. CRSI is simpler and more proven than
standard RSI for mean reversion. Dual HTF (1d+1w) provides stronger trend filter than single HTF.
Relaxed CRSI thresholds ensure we generate enough trades (avoid 0-trade failure seen in 40+ strategies).
12h TF naturally targets 20-50 trades/year which minimizes fee drag.

Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crsi_chop_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    if half < 1 or sqrt_n < 1:
        return hma
    
    # WMA helper
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    
    return hma

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi3 = 100.0 - (100.0 / (1.0 + rs))
    rsi3 = np.concatenate([[np.nan], rsi3[1:]])
    
    # Streak RSI(2)
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
    
    streak_gain_s = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_s = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_gain_s / (streak_loss_s + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = np.where(np.isnan(rsi_streak), 50.0, rsi_streak)
    
    # PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i - rank_period:i + 1])
        if len(returns) > 0 and not np.all(np.isnan(returns)):
            current_return = returns[-1]
            rank = np.sum(returns[:-1] < current_return) / (len(returns) - 1)
            percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    for i in range(n):
        if not np.isnan(rsi3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/chop (mean reversion regime)
    CHOP < 38.2 = trending (trend follow regime)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        if tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    hma_12h = calculate_hma(close, period=21)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(hma_12h[i]):
            continue
        if np.isnan(crsi_12h[i]):
            continue
        if np.isnan(chop_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop = chop_12h[i] > 61.8  # Range/mean reversion regime
        is_trend = chop_12h[i] < 38.2  # Trending regime
        
        # === HTF MAJOR TREND BIAS (1d + 1w HMA) ===
        # Both HTF must align for strong signal
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong HTF alignment
        htf_strong_bull = htf_1d_bullish and htf_1w_bullish
        htf_strong_bear = htf_1d_bearish and htf_1w_bearish
        
        # === PRIMARY TREND (12h HMA) ===
        price_above_hma = close[i] > hma_12h[i]
        price_below_hma = close[i] < hma_12h[i]
        
        # HMA slope (5 bar lookback)
        hma_slope_up = hma_12h[i] > hma_12h[i - 5] if i >= 5 else False
        hma_slope_down = hma_12h[i] < hma_12h[i - 5] if i >= 5 else False
        
        # === CRSI SIGNALS (relaxed thresholds for trade generation) ===
        crsi_extreme_oversold = crsi_12h[i] < 15.0  # Relaxed from 10
        crsi_extreme_overbought = crsi_12h[i] > 85.0  # Relaxed from 90
        crsi_oversold = crsi_12h[i] < 30.0
        crsi_overbought = crsi_12h[i] > 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        long_confidence = 0
        
        # HTF alignment (strongest signal when both 1d and 1w bullish)
        if htf_strong_bull:
            long_confidence += 3
        elif htf_1d_bullish or htf_1w_bullish:
            long_confidence += 1
        
        # Primary trend alignment
        if price_above_hma:
            long_confidence += 1
        
        # HMA slope confirmation
        if hma_slope_up:
            long_confidence += 1
        
        # CRSI entry (different logic per regime)
        if is_chop:
            # Range regime: need extreme CRSI for mean reversion
            if crsi_extreme_oversold:
                long_confidence += 3
            elif crsi_oversold:
                long_confidence += 1
        elif is_trend:
            # Trend regime: moderate CRSI pullback OK
            if crsi_oversold:
                long_confidence += 2
        else:
            # Neutral regime
            if crsi_extreme_oversold:
                long_confidence += 2
            elif crsi_oversold:
                long_confidence += 1
        
        # Enter long if confidence >= 5 (relaxed for trade generation)
        if long_confidence >= 5:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_confidence = 0
            
            # HTF alignment
            if htf_strong_bear:
                short_confidence += 3
            elif htf_1d_bearish or htf_1w_bearish:
                short_confidence += 1
            
            # Primary trend
            if price_below_hma:
                short_confidence += 1
            
            # HMA slope
            if hma_slope_down:
                short_confidence += 1
            
            # CRSI entry
            if is_chop:
                if crsi_extreme_overbought:
                    short_confidence += 3
                elif crsi_overbought:
                    short_confidence += 1
            elif is_trend:
                if crsi_overbought:
                    short_confidence += 2
            else:
                if crsi_extreme_overbought:
                    short_confidence += 2
                elif crsi_overbought:
                    short_confidence += 1
            
            if short_confidence >= 5:
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma and htf_1d_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and price_below_hma and htf_1d_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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