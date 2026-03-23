#!/usr/bin/env python3
"""
Experiment #661: 4h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + Donchian

Hypothesis: 4h timeframe with regime-switching logic adapts to market conditions better
than single-strategy approaches. Choppiness Index cleanly separates choppy vs trending
markets. Connors RSI excels at mean reversion in ranges (75% win rate documented).
Donchian breakouts capture trends when CHOP confirms trending regime.

Key innovations:
1. Choppiness Index (14) regime: >55 = chop (mean revert), <45 = trend (breakout)
2. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — superior to standard RSI
3. Donchian(20) breakout for trend entries with HMA(21) confirmation
4. 1d HMA for intermediate trend bias, 1w HMA for macro bias
5. ATR(14) trailing stop at 2.5x for risk management
6. Looser CRSI thresholds (15/85 instead of 10/90) to ensure trade frequency

Why this should beat Sharpe=0.612:
- Regime-switching adapts to 2022 crash (chop) and 2021 bull (trend)
- Connors RSI has documented edge in mean reversion (Research Report #4)
- 4h TF = 20-50 trades/year target, optimal fee/trade balance
- Dual HTF (1d + 1w) provides layered trend confirmation
- Conservative sizing (0.28) survives 77% crash with ~27% DD

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_donchian_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) — combines 3 components for mean reversion signals.
    
    Components:
    1. RSI(3) — short-term momentum
    2. RSI of streak length(2) — consecutive up/down days
    3. Percent Rank(100) — where current price ranks vs last 100 days
    
    CRSI = (RSI_close + RSI_streak + PercentRank) / 3
    
    Long: CRSI < 15 (oversold)
    Short: CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Component 2: RSI of streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of absolute streak values
    streak_abs = np.abs(streak)
    streak_delta = np.diff(streak_abs, prepend=streak_abs[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: Percent Rank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / pr_period * 100
        percent_rank[i] = rank
    
    # Combine components
    with np.errstate(invalid='ignore'):
        crsi = (rsi_close + rsi_streak + percent_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    
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
    """Donchian Channel — highest high and lowest low over period."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother trend detection."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    crsi_4h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    donchian_upper_4h, donchian_lower_4h, donchian_mid_4h = calculate_donchian(high, low, period=20)
    hma_4h = calculate_hma(close, period=21)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.28
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(donchian_upper_4h[i]) or np.isnan(hma_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_4h[i] > 55.0
        is_trending = chop_4h[i] < 45.0
        
        # === HTF TREND BIAS ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === 4h TREND (HMA) ===
        hma_bullish = close[i] > hma_4h[i]
        hma_bearish = close[i] < hma_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper_4h[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower_4h[i-1]  # Break below previous lower
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 15  # Looser threshold for more trades
        crsi_overbought = crsi_4h[i] > 85
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with Connors RSI) ===
        if is_choppy:
            # Long: CRSI oversold + HTF 1d not strongly bearish
            if crsi_oversold and not htf_1d_bearish:
                desired_signal = SIZE_LONG
            # Short: CRSI overbought + HTF 1d not strongly bullish
            elif crsi_overbought and not htf_1d_bullish:
                desired_signal = -SIZE_SHORT
            # CRSI extreme with 1w confirmation
            elif crsi_oversold and htf_1w_bullish:
                desired_signal = SIZE_LONG
            elif crsi_overbought and htf_1w_bearish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Breakout with Donchian + HMA) ===
        elif is_trending:
            # Long: Donchian breakout + HMA bullish + HTF supportive
            if donchian_breakout_long and hma_bullish and htf_1d_bullish:
                desired_signal = SIZE_LONG
            # Short: Donchian breakout + HMA bearish + HTF supportive
            elif donchian_breakout_short and hma_bearish and htf_1d_bearish:
                desired_signal = -SIZE_SHORT
            # HMA cross with trend confirmation
            elif hma_bullish and htf_1w_bullish and crsi_4h[i] < 50:
                desired_signal = SIZE_LONG
            elif hma_bearish and htf_1w_bearish and crsi_4h[i] > 50:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use HMA direction with CRSI filter
            if hma_bullish and crsi_oversold:
                desired_signal = SIZE_LONG
            elif hma_bearish and crsi_overbought:
                desired_signal = -SIZE_SHORT
            # Donchian breakout with single HTF confirmation
            elif donchian_breakout_long and htf_1d_bullish:
                desired_signal = SIZE_LONG
            elif donchian_breakout_short and htf_1d_bearish:
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
            if position_side > 0:
                # Hold long if HMA still bullish OR CRSI not extremely overbought
                if hma_bullish and crsi_4h[i] < 80:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HMA still bearish OR CRSI not extremely oversold
                if hma_bearish and crsi_4h[i] > 20:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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