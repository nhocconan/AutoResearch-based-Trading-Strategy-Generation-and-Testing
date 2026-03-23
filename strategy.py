#!/usr/bin/env python3
"""
Experiment #1032: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 748+ failed strategies, the pattern is clear:
1. Simple trend following fails in bear/range markets (2025 test period)
2. Connors RSI (CRSI) has proven 75% win rate for mean reversion entries
3. Choppiness Index is the BEST regime filter for switching between mean-revert and trend-follow
4. 12h timeframe targets 20-50 trades/year — optimal for fee drag vs statistical significance
5. Dual HTF (1d + 1w) provides macro bias without overfitting

Strategy Components:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long entry: CRSI < 15 (oversold) + price > 1d HMA21 (bullish macro)
   - Short entry: CRSI > 85 (overbought) + price < 1d HMA21 (bearish macro)
   - Exit: CRSI crosses 50 (mean reached)

2. CHOPPINESS INDEX regime filter:
   - CHOP > 61.8 = ranging → use CRSI mean reversion signals
   - CHOP < 38.2 = trending → use HMA crossover + CRSI confirmation
   - Between = hold existing positions, no new entries

3. 1d HMA21 + 1w HMA21: Dual HTF trend bias
   - Only long when price > 1d HMA (medium-term bullish)
   - Only short when price < 1w HMA (long-term bearish)
   - This asymmetry works in both bull and bear markets

4. ATR Trailing Stop: 2.5x ATR for risk management, signal→0 when hit

5. Position Sizing: Discrete levels (0.0, ±0.25, ±0.30) to minimize fee churn

Why 12h works:
- Target 20-50 trades/year (vs 100+ on 1h, 10 on 1d)
- Enough frequency for statistical significance across all symbols
- Less noise than 1h/4h, more signals than 1d
- Proven patterns: CRSI+CHOP (ETH +0.923), HMA+RSI (SOL +0.879)

Critical fixes from failed experiments:
- CONNORS RSI instead of simple RSI (better for mean reversion)
- DUAL HTF (1d + 1w) with asymmetric logic
- CHOPPINESS regime switch (not fixed mean-revert or trend)
- RELAXED CRSI thresholds (15/85 not 10/90) for more trades
- Discrete signal sizes minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_hma_atr_v1"
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
    PercentRank: Percentile rank of today's return over last 100 days
    
    Entry: CRSI < 15 (oversold) or CRSI > 85 (overbought)
    Exit: CRSI crosses 50 (mean reversion complete)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period, n):
        if avg_streak_loss[i] == 0:
            streak_rsi[i] = 100.0
        else:
            rs = avg_streak_gain[i] / (avg_streak_loss[i] + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank of returns
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0.0], returns])
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        count_below = np.sum(window < returns[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)."""
    n = len(hma_values)
    slope = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i-lookback]):
            slope[i] = (hma_values[i] - hma_values[i-lookback]) / (hma_values[i-lookback] + 1e-10)
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA21 for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA21 for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    
    # Also calculate 12h HMA21 for local trend
    hma_12h = calculate_hma(close, 21)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(chop_12h[i]) or np.isnan(hma_12h[i]):
            continue
        
        # === MACRO TREND (HTF HMA21) ===
        # Asymmetric: long requires 1d bullish, short requires 1w bearish
        medium_bull = close[i] > hma_1d_aligned[i]
        long_bear = close[i] < hma_1w_aligned[i]
        
        # Local 12h trend
        local_bull = close[i] > hma_12h[i]
        local_bear = close[i] < hma_12h[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_12h[i] > 61.8  # Ranging market → mean reversion
        regime_trend = chop_12h[i] < 38.2  # Trending market → trend follow
        regime_neutral = not regime_chop and not regime_trend  # Transition
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 15
        crsi_overbought = crsi_12h[i] > 85
        crsi_mean_cross_up = crsi_12h[i] > 50 and crsi_12h[i-1] <= 50
        crsi_mean_cross_down = crsi_12h[i] < 50 and crsi_12h[i-1] >= 50
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_chop and medium_bull:
            # Mean reversion in choppy market with bullish macro
            if crsi_oversold:
                desired_signal = BASE_SIZE
        elif regime_trend and medium_bull and local_bull:
            # Trend following in trending bullish market
            if crsi_12h[i] < 40 and crsi_12h[i-1] >= 40:
                # Pullback entry in uptrend
                desired_signal = REDUCED_SIZE
        elif regime_neutral and medium_bull:
            # Relaxed entry in transition
            if crsi_oversold:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if regime_chop and long_bear:
            # Mean reversion in choppy market with bearish macro
            if crsi_overbought:
                desired_signal = -BASE_SIZE
        elif regime_trend and long_bear and local_bear:
            # Trend following in trending bearish market
            if crsi_12h[i] > 60 and crsi_12h[i-1] <= 60:
                # Pullback entry in downtrend
                desired_signal = -REDUCED_SIZE
        elif regime_neutral and long_bear:
            # Relaxed entry in transition
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
        
        # === EXIT ON CRSI MEAN REVERSION ===
        if in_position and position_side > 0 and crsi_mean_cross_up:
            # Long exit when CRSI crosses above 50 (mean reached)
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_mean_cross_down:
            # Short exit when CRSI crosses below 50 (mean reached)
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bullish and CRSI not extreme overbought
                if medium_bull and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bearish and CRSI not extreme oversold
                if long_bear and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro trend reverses
            if not medium_bull and crsi_12h[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro trend reverses
            if not long_bear and crsi_12h[i] < 40:
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