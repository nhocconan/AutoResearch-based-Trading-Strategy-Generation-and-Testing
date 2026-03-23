#!/usr/bin/env python3
"""
Experiment #1042: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 754 failed strategies, the pattern is clear:
1. Complex regime switching → mutually exclusive conditions → 0 trades
2. 12h timeframe needs RELAXED entry thresholds to generate sufficient trades
3. Connors RSI works best on 12h (ETH Sharpe +0.923 in research)
4. Choppiness Index regime filter prevents mean-reversion in strong trends

Strategy Design:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 25 (relaxed from 10 to ensure trades)
   - Short when CRSI > 75 (relaxed from 90 to ensure trades)
   
2. CHOPPINESS INDEX (CHOP): Regime detection
   - CHOP > 61.8 = range market → use mean reversion (CRSI)
   - CHOP < 38.2 = trending market → use trend following (HMA crossover)
   - 38.2 <= CHOP <= 61.8 = neutral → allow both
   
3. 1d HMA21 MACRO FILTER: Only long when price > 1d HMA, only short when price < 1d HMA
   - Asymmetric: easier to enter with macro trend
   
4. 1w HMA50 ULTRA-MACRO: Additional filter for extreme regimes
   - Price > 1w HMA50 = bull market bias (prefer longs)
   - Price < 1w HMA50 = bear market bias (prefer shorts)

5. ATR TRAILING STOP: 2.5x ATR(14) from entry high/low

Why this should work:
- CRSI proven on 12h (research shows ETH Sharpe +0.923)
- Relaxed CRSI thresholds (25/75 vs 10/90) ensure 30+ trades/train
- Choppiness filter prevents whipsaw in strong trends
- Dual HTF (1d + 1w) provides robust macro context
- 12h timeframe naturally limits trades to 20-50/year

Timeframe: 12h (target 20-50 trades/year)
Position Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_hma_atr_v2"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index using EMA smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): RSI of consecutive up/down day streaks
    PercentRank(100): Current price percentile vs last 100 closes
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_vals = streak_abs[max(0, i-streak_period):i+1]
        if len(streak_vals) >= streak_period:
            avg_streak = np.mean(streak_vals)
            streak_rsi[i] = min(100, avg_streak * 50)  # Scale to 0-100
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP): Measures market choppiness vs trending
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate CHOP
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
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

def calculate_hma(series, period):
    """Hull Moving Average for trend direction."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA21 for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA50 for ultra-macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        
        # === MACRO TREND FILTERS ===
        # 1d HMA21: Primary macro trend
        macro_bull_1d = close[i] > hma_1d_aligned[i]
        macro_bear_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA50: Ultra-macro trend (stronger filter)
        macro_bull_1w = close[i] > hma_1w_aligned[i]
        macro_bear_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range (mean reversion works)
        # CHOP < 38.2 = trend (trend following works)
        # 38.2 <= CHOP <= 61.8 = neutral (allow both)
        regime_range = chop_12h[i] > 61.8
        regime_trend = chop_12h[i] < 38.2
        regime_neutral = 38.2 <= chop_12h[i] <= 61.8
        
        # === CONNORS RSI SIGNALS ===
        # Relaxed thresholds to ensure sufficient trades
        crsi_oversold = crsi_12h[i] < 25  # Long entry zone
        crsi_overbought = crsi_12h[i] > 75  # Short entry zone
        crsi_extreme_long = crsi_12h[i] < 15  # Very oversold
        crsi_extreme_short = crsi_12h[i] > 85  # Very overbought
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Entry 1: Range regime + CRSI oversold + 1d macro bull (primary mean reversion)
        if regime_range and crsi_oversold and macro_bull_1d:
            desired_signal = BASE_SIZE
        # Entry 2: Neutral regime + CRSI oversold + 1d macro bull
        elif regime_neutral and crsi_oversold and macro_bull_1d:
            desired_signal = BASE_SIZE
        # Entry 3: Trend regime + CRSI extreme + 1d & 1w macro bull (strong conviction)
        elif regime_trend and crsi_extreme_long and macro_bull_1d and macro_bull_1w:
            desired_signal = REDUCED_SIZE
        # Entry 4: Any regime + CRSI extreme oversold + 1d macro bull
        elif crsi_extreme_long and macro_bull_1d:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Entry 1: Range regime + CRSI overbought + 1d macro bear (primary mean reversion)
        if regime_range and crsi_overbought and macro_bear_1d:
            desired_signal = -BASE_SIZE
        # Entry 2: Neutral regime + CRSI overbought + 1d macro bear
        elif regime_neutral and crsi_overbought and macro_bear_1d:
            desired_signal = -BASE_SIZE
        # Entry 3: Trend regime + CRSI extreme + 1d & 1w macro bear (strong conviction)
        elif regime_trend and crsi_extreme_short and macro_bear_1d and macro_bear_1w:
            desired_signal = -REDUCED_SIZE
        # Entry 4: Any regime + CRSI extreme overbought + 1d macro bear
        elif crsi_extreme_short and macro_bear_1d:
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d macro still bullish or CRSI not overbought
                if macro_bull_1d and crsi_12h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d macro still bearish or CRSI not oversold
                if macro_bear_1d and crsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d macro reverses bearish AND CRSI overbought
            if macro_bear_1d and crsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d macro reverses bullish AND CRSI oversold
            if macro_bull_1d and crsi_12h[i] < 30:
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