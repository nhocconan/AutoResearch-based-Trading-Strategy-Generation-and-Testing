#!/usr/bin/env python3
"""
Experiment #641: 4h Primary + 1d/1w HTF — Regime-Adaptive CRSI + Donchian + HMA

Hypothesis: 4h timeframe with daily/weekly HTF filter provides optimal balance between
signal quality and trade frequency. Choppiness Index detects regime, then we switch
between mean reversion (CRSI extremes in chop) and trend follow (Donchian breakout in trend).

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven 75% win rate
2. Choppiness Index regime: CHOP>55 = mean revert, CHOP<45 = trend follow
3. Donchian(20) breakout for trend entries with HMA(21) confirmation
4. 1d HMA for macro bias — only trade with daily trend
5. 1w HMA for ultimate filter — avoid counter-trend vs weekly
6. Simple hold logic — maintain position while regime unchanged
7. ATR(14) trailing stop at 2.5x for risk management

Why this should beat Sharpe=0.612:
- CRSI has documented edge in mean reversion (better than simple RSI)
- Regime switching avoids whipsaw in choppy markets
- 4h TF = 20-50 trades/year target (optimal fee vs signal ratio)
- 1d/1w HTF prevents major counter-trend disasters (2022 crash, 2025 bear)
- Conservative sizing (0.30) survives 77% crash with ~27% DD

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_crsi_donchian_hma_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) of close — short-term momentum
    2. RSI(2) of streak — streak duration (consecutive up/down days)
    3. PercentRank(100) — where current close ranks vs last 100 closes
    
    Long signal: CRSI < 10 (oversold)
    Short signal: CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) of close
    def calc_rsi(prices, period):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi = np.nan_to_num(rsi, nan=50.0)
        rsi = np.concatenate([[50.0] * period, rsi])
        return rsi
    
    rsi_close = calc_rsi(close, rsi_period)
    
    # Streak calculation (consecutive up/down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) of streak
    rsi_streak = calc_rsi(streak, streak_period)
    
    # PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/ranging
    CHOP < 38.2 = trending
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
    Upper = Highest High over period
    Lower = Lowest Low over period
    Middle = (Upper + Lower) / 2
    
    Breakout above upper = long signal
    Breakdown below lower = short signal
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
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
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size (30% of capital)
    
    # Position tracking for stoploss
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_4h[i] < 15.0
        crsi_overbought = crsi_4h[i] > 85.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with CRSI) ===
        if is_choppy:
            # Long: CRSI oversold + HTF 1d not strongly bearish
            if crsi_oversold and not htf_1d_bearish:
                desired_signal = SIZE
            # Short: CRSI overbought + HTF 1d not strongly bullish
            elif crsi_overbought and not htf_1d_bullish:
                desired_signal = -SIZE
        
        # === REGIME 2: TRENDING MARKET (Trend Follow with Donchian) ===
        elif is_trending:
            # Long: Donchian breakout + HTF 1d bullish + HTF 1w not bearish
            if donchian_breakout_long and htf_1d_bullish and not htf_1w_bearish:
                desired_signal = SIZE
            # Short: Donchian breakdown + HTF 1d bearish + HTF 1w not bullish
            elif donchian_breakout_short and htf_1d_bearish and not htf_1w_bullish:
                desired_signal = -SIZE
        
        # === REGIME 3: NEUTRAL/TRANSITION (Use HTF trend) ===
        else:
            # Follow 1d trend with CRSI filter
            if htf_1d_bullish and crsi_4h[i] < 60:
                desired_signal = SIZE
            elif htf_1d_bearish and crsi_4h[i] > 40:
                desired_signal = -SIZE
        
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
        # This prevents premature exits and ensures adequate trade frequency
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d still bullish OR CRSI not extremely overbought
                if htf_1d_bullish and crsi_4h[i] < 90:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if 1d still bearish OR CRSI not extremely oversold
                if htf_1d_bearish and crsi_4h[i] > 10:
                    desired_signal = -SIZE
        
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