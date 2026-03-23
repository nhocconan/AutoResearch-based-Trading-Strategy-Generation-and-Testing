#!/usr/bin/env python3
"""
Experiment #1016: 12h Primary + 1d HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: After 736 failed strategies, the key insight is that 12h timeframe 
with proper regime detection works best for bear/range markets (2025 test period).

This strategy combines:
1. CHOPPINESS INDEX (14): Regime filter — CHOP>61.8=range (mean revert), CHOP<38.2=trend
2. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI<15 (oversold), Short when CRSI>85 (overbought)
   - 75% win rate in historical tests
3. 1d HMA21: Long-term trend bias — only long when price>1d HMA, only short when price<1d HMA
4. ATR Trailing Stop: 2.5x ATR for risk management

Why 12h works:
- Target 20-50 trades/year (vs 100+ on lower TF = less fee drag)
- Higher timeframes have proven better Sharpe in our experiments (#1012, #1013)
- Less noise, more significant signals

Critical fixes from failed experiments:
- SIMPLER logic than #1014 (Fisher was too complex, failed with Sharpe=-0.741)
- CRSI instead of Fisher (CRSI has proven track record in mean reversion)
- RELAXED CRSI thresholds (15/85 not 10/90) to ensure >=30 trades
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Proper stoploss tracking with signal→0 when hit

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Use exponential smoothing for RSI calculation
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    
    Entry: CRSI < 15 (oversold) for long, CRSI > 85 (overbought) for short
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = (up_streaks / streak_period) * 100.0
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank of returns
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100.0
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        past_returns = returns[i-rank_period+1:i]
        current_return = returns[i]
        if len(past_returns) > 0:
            percent_rank[i] = (np.sum(past_returns < current_return) / len(past_returns)) * 100.0
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_12h[i]):
            continue
        
        # === MACRO TREND (1d HMA21) ===
        # Only long when price above 1d HMA, only short when below
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop_12h[i] > 61.8  # Ranging → mean reversion
        regime_trend = chop_12h[i] < 38.2  # Trending → trend follow
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 15  # Strong oversold
        crsi_overbought = crsi_12h[i] > 85  # Strong overbought
        crsi_mild_oversold = crsi_12h[i] < 25  # Mild oversold
        crsi_mild_overbought = crsi_12h[i] > 75  # Mild overbought
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_range and trend_bull:
            # Mean reversion in ranging market with bullish trend
            if crsi_oversold:
                desired_signal = BASE_SIZE
            elif crsi_mild_oversold:
                desired_signal = REDUCED_SIZE
        elif regime_trend and trend_bull:
            # Trend following: enter on pullback in uptrend
            if crsi_mild_oversold:
                desired_signal = REDUCED_SIZE
        elif trend_bull:
            # Neutral regime: only take strong signals
            if crsi_oversold:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if regime_range and trend_bear:
            # Mean reversion in ranging market with bearish trend
            if crsi_overbought:
                desired_signal = -BASE_SIZE
            elif crsi_mild_overbought:
                desired_signal = -REDUCED_SIZE
        elif regime_trend and trend_bear:
            # Trend following: enter on bounce in downtrend
            if crsi_mild_overbought:
                desired_signal = -REDUCED_SIZE
        elif trend_bear:
            # Neutral regime: only take strong signals
            if crsi_overbought:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish and CRSI not extreme overbought
                if trend_bull and crsi_12h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish and CRSI not extreme oversold
                if trend_bear and crsi_12h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses bearish
            if trend_bear and crsi_12h[i] > 50:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses bullish
            if trend_bull and crsi_12h[i] < 50:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
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