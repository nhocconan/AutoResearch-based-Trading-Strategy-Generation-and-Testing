#!/usr/bin/env python3
"""
Experiment #662: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness + Donchian

Hypothesis: 12h timeframe with Connors RSI (proven 75% win rate) + Choppiness regime 
filter + Donchian breakout confirmation will generate 20-50 trades/year with positive 
Sharpe across all symbols. Connors RSI excels at catching reversals in bear/range 
markets (2022, 2025) while Donchian provides trend breakout confirmation.

Key innovations:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long entry: CRSI < 15 (oversold) + price > SMA200
   - Short entry: CRSI > 85 (overbought) + price < SMA200
2. Choppiness Index regime switch: CHOP > 55 = mean revert, CHOP < 45 = trend follow
3. Donchian(20) breakout confirmation for trend entries
4. 1d HMA for intermediate trend, 1w HMA for macro bias
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete signal sizes: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should beat Sharpe=0.612:
- 12h timeframe = natural 20-50 trades/year (proven in exp #652)
- Connors RSI has documented edge in bear markets (2022 crash)
- Choppiness filter prevents trend strategies in chop (major loss source)
- Dual HTF (1d + 1w) provides cleaner trend bias than single HTF
- Conservative sizing (0.25-0.30) survives 77% crash with ~25% DD

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_donchian_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): Streak duration RSI (consecutive up/down days)
    3. PercentRank(100): Where current price ranks vs last 100 periods
    
    Long signal: CRSI < 15 (extreme oversold)
    Short signal: CRSI > 85 (extreme overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    def calc_rsi(prices, period):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
        
        # Pad first value
        rsi = np.concatenate([[50], rsi])
        return rsi
    
    rsi_short = calc_rsi(close, rsi_period)
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = calc_rsi(streak, streak_period)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1)
        percent_rank[i] = rank * 100
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 55 = chop (mean revert), < 45 = trend (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Sum ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel.
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = bullish, below lower = bearish
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_sma(close, period=200):
    """Simple Moving Average for trend filter."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start later to ensure all indicators ready (CRSI needs 100)
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]):
            continue
        if np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_12h[i] > 55.0
        is_trending = chop_12h[i] < 45.0
        
        # === HTF TREND BIAS ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === SMA 200 TREND FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 15.0
        crsi_overbought = crsi_12h[i] > 85.0
        crsi_neutral_long = crsi_12h[i] < 30.0
        crsi_neutral_short = crsi_12h[i] > 70.0
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with CRSI) ===
        if is_choppy:
            # Long: CRSI oversold + above SMA200 (uptrend pullback)
            if crsi_oversold and above_sma200:
                desired_signal = SIZE_LONG
            # Short: CRSI overbought + below SMA200 (downtrend rally)
            elif crsi_overbought and below_sma200:
                desired_signal = -SIZE_SHORT
            # Less strict CRSI entries with HTF confirmation
            elif crsi_neutral_long and htf_1d_bullish and htf_1w_bullish:
                desired_signal = SIZE_LONG
            elif crsi_neutral_short and htf_1d_bearish and htf_1w_bearish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Follow with Donchian + CRSI) ===
        elif is_trending:
            # Long: Donchian breakout + CRSI not overbought + HTF bullish
            if donchian_breakout_long and crsi_12h[i] < 70.0 and htf_1d_bullish:
                desired_signal = SIZE_LONG
            # Short: Donchian breakdown + CRSI not oversold + HTF bearish
            elif donchian_breakout_short and crsi_12h[i] > 30.0 and htf_1d_bearish:
                desired_signal = -SIZE_SHORT
            # Trend continuation with CRSI pullback
            elif htf_1d_bullish and htf_1w_bullish and crsi_neutral_long:
                desired_signal = SIZE_LONG
            elif htf_1d_bearish and htf_1w_bearish and crsi_neutral_short:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use HTF direction with CRSI filter
            if htf_1d_bullish and crsi_neutral_long:
                desired_signal = SIZE_LONG
            elif htf_1d_bearish and crsi_neutral_short:
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
        
        # === EXIT CONDITIONS (CRSI extreme reversal) ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Exit long if CRSI becomes overbought
                if crsi_12h[i] > 80.0:
                    desired_signal = 0.0
                # Hold if HTF still bullish and CRSI reasonable
                elif htf_1d_bullish and crsi_12h[i] < 75.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Exit short if CRSI becomes oversold
                if crsi_12h[i] < 20.0:
                    desired_signal = 0.0
                # Hold if HTF still bearish and CRSI reasonable
                elif htf_1d_bearish and crsi_12h[i] > 25.0:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            desired_signal = SIZE_LONG
        elif desired_signal < -0.15:
            desired_signal = -SIZE_SHORT
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
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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