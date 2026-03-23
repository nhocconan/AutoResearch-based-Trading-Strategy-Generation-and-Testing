#!/usr/bin/env python3
"""
Experiment #862: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime

Hypothesis: Exp#852 showed 12h CRSI+Choppiness achieved Sharpe=0.504. This experiment
improves on it by:
1. Using Connors RSI (CRSI) instead of regular RSI — proven 75% win rate for reversals
2. Dual HTF trend filter (1d HMA21 + 1w HMA21) for stronger bias confirmation
3. Relaxed CRSI thresholds (15/85 instead of 10/90) to ensure sufficient trades
4. ATR trailing stop at 2.5x to protect against 2022-style crashes
5. Discrete position sizing (0.0, ±0.20, ±0.30) to minimize fee churn

CRSI Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(3): Short-term momentum
- RSI_Streak(2): Consecutive up/down days
- PercentRank(100): Where current close ranks vs last 100 closes

Regime Logic:
- CHOP > 55: Range regime → Mean reversion (CRSI extremes)
- CHOP < 45: Trend regime → Pullback entries (CRSI mid-range + HTF trend)
- 45-55: Neutral → Conservative entries only

Target: Sharpe > 0.612, trades >= 20 train, >= 5 test per symbol
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_dual_hma_1d1w_atr_v2"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Range: 0-100. Extreme < 10 = oversold, > 90 = overbought.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak = 0
        if i > 0:
            if close[i] > close[i-1]:
                streak = 1
                j = i - 1
                while j > 0 and close[j] > close[j-1]:
                    streak += 1
                    j -= 1
            elif close[i] < close[i-1]:
                streak = -1
                j = i - 1
                while j > 0 and close[j] < close[j-1]:
                    streak -= 1
                    j -= 1
        
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            streak_rsi[i] = 100 * streak / (streak + 1) if streak > 0 else 50
        else:
            streak_rsi[i] = 100 * (abs(streak) + 1) / (abs(streak) + 2) if streak < 0 else 50
    
    # Component 3: Percent Rank (where close ranks vs last 100 closes)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100 * rank / (rank_period - 1)
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[i-1]))
    
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
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF TREND BIAS (1d + 1w HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend confirmation (both HTF agree)
        strong_bullish = trend_1d_bullish and trend_1w_bullish
        strong_bearish = trend_1d_bearish and trend_1w_bearish
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        
        # === CRSI SIGNALS (Relaxed for more trades) ===
        crsi_oversold = crsi_12h[i] < 20
        crsi_overbought = crsi_12h[i] > 80
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        crsi_moderate_oversold = 20 <= crsi_12h[i] < 35
        crsi_moderate_overbought = 65 < crsi_12h[i] <= 80
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + at least one HTF bullish
            if crsi_oversold and (trend_1d_bullish or trend_1w_bullish):
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + at least one HTF bearish
            if crsi_overbought and (trend_1d_bearish or trend_1w_bearish):
                desired_signal = -BASE_SIZE
            
            # Extreme CRSI alone (ensures trades on all symbols)
            if crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
            
            # Moderate CRSI + strong HTF agreement
            if crsi_moderate_oversold and strong_bullish and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_moderate_overbought and strong_bearish and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Strong bullish trend + CRSI pullback
            if strong_bullish:
                if crsi_moderate_oversold or crsi_oversold:
                    desired_signal = BASE_SIZE
            
            # Short: Strong bearish trend + CRSI pullback
            if strong_bearish:
                if crsi_moderate_overbought or crsi_overbought:
                    desired_signal = -BASE_SIZE
            
            # Single HTF trend + extreme CRSI
            if trend_1d_bullish and crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and crsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Extreme CRSI + any HTF alignment
            if crsi_extreme_oversold and (trend_1d_bullish or trend_1w_bullish):
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (trend_1d_bearish or trend_1w_bearish):
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
                # Hold long if HTF trend intact and CRSI not overbought
                if (trend_1d_bullish or trend_1w_bullish) and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend intact and CRSI not oversold
                if (trend_1d_bearish or trend_1w_bearish) and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HTF turn bearish + CRSI overbought
            if strong_bearish and crsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HTF turn bullish + CRSI oversold
            if strong_bullish and crsi_12h[i] < 20:
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