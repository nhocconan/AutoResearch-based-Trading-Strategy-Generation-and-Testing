#!/usr/bin/env python3
"""
Experiment #636: 12h Primary + 1d HTF — Connors RSI + Donchian + Choppiness Regime

Hypothesis: 12h timeframe with 1d HTF filter balances signal quality with trade frequency.
Connors RSI (CRSI) has documented 75% win rate for mean reversion entries. Combined with
Donchian breakout for trend confirmation and Choppiness Index for regime detection, this
should work in both bull (2021), bear (2022), and range (2025) markets.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 (oversold pullback in uptrend)
   - Short: CRSI > 85 (overbought rally in downtrend)
2. Donchian(20) breakout confirmation — ensures momentum backing the entry
3. Choppiness Index regime switch — mean revert when CHOP>55, trend follow when CHOP<45
4. 1d HMA for macro bias — only long when daily trend supportive
5. ATR trailing stop (2.5x) — protects capital in adverse moves
6. Hold logic — maintain position if Donchian trend intact (reduces churn)

Why this should beat Sharpe=0.612:
- CRSI proven edge in bear/range markets (2022 crash, 2025 bear)
- 12h timeframe = fewer false signals than 4h, more trades than 1d
- 1d HTF filter prevents counter-trend trades in strong macro moves
- Donchian confirmation ensures we're not catching falling knives
- Conservative sizing (0.28) survives 77% crash with ~27% DD

Target: Sharpe > 0.612, trades >= 20 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_donchian_chop_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) — Larry Connors' composite mean reversion indicator.
    
    Formula:
    1. RSI(close, 3) — short-term momentum
    2. RSI(streak, 2) — streak duration (consecutive up/down days)
    3. PercentRank(close, 100) — where current close ranks vs last 100 closes
    
    CRSI = (RSI_3 + RSI_Streak_2 + PercentRank) / 3
    
    Long signal: CRSI < 15 (extreme oversold)
    Short signal: CRSI > 85 (extreme overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period:
        return crsi
    
    # RSI(3)
    def calc_rsi(series, period):
        delta = np.diff(series)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi = np.where(avg_loss == 0, 100, rsi)
        
        # Pad to match length
        rsi = np.concatenate([[np.nan] * (period - 1), rsi])
        return rsi
    
    rsi_3 = calc_rsi(close, rsi_period)
    
    # Streak RSI — count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to absolute values for RSI calculation
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    
    # RSI on streak (up streaks = gains, down streaks = losses)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streaks = np.sum((streak[i-streak_period+1:i+1] > 0).astype(float))
        down_streaks = np.sum((streak[i-streak_period+1:i+1] < 0).astype(float))
        
        if down_streaks == 0:
            streak_rsi[i] = 100
        else:
            rs = up_streaks / down_streaks
            streak_rsi[i] = 100 - (100 / (1 + rs))
    
    # PercentRank — where current close ranks vs last 100 closes
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100 * rank / pr_period
    
    # Combine into CRSI
    for i in range(pr_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout detection.
    Upper = highest high over period
    Lower = lowest low over period
    
    Long breakout: close crosses above upper
    Short breakout: close crosses below lower
    """
    n = len(close)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(crsi_12h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_12h[i] > 55.0
        is_trending = chop_12h[i] < 45.0
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN TREND ===
        donchian_bullish = close[i] > donchian_mid[i]
        donchian_bearish = close[i] < donchian_mid[i]
        
        # Donchian breakout signals
        donchian_long_breakout = close[i] > donchian_upper[i]
        donchian_short_breakout = close[i] < donchian_lower[i]
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 15.0
        crsi_overbought = crsi_12h[i] > 85.0
        
        # CRSI moderate levels for trend continuation
        crsi_bullish = crsi_12h[i] < 50.0
        crsi_bearish = crsi_12h[i] > 50.0
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with CRSI) ===
        if is_choppy:
            # Long: CRSI oversold + HTF 1d not strongly bearish
            if crsi_oversold and not htf_1d_bearish:
                desired_signal = SIZE_LONG
            # Short: CRSI overbought + HTF 1d not strongly bullish
            elif crsi_overbought and not htf_1d_bullish:
                desired_signal = -SIZE_SHORT
            # Donchian mean reversion (fade breakouts in chop)
            elif donchian_short_breakout and crsi_12h[i] > 70:
                desired_signal = -SIZE_SHORT
            elif donchian_long_breakout and crsi_12h[i] < 30:
                desired_signal = SIZE_LONG
        
        # === REGIME 2: TRENDING MARKET (Trend Follow with Donchian + CRSI) ===
        elif is_trending:
            # Long: HTF bullish + Donchian bullish + CRSI not overbought
            if htf_1d_bullish and donchian_bullish and crsi_bullish:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + Donchian bearish + CRSI not oversold
            elif htf_1d_bearish and donchian_bearish and crsi_bearish:
                desired_signal = -SIZE_SHORT
            # Donchian breakout with trend confirmation
            elif donchian_long_breakout and htf_1d_bullish:
                desired_signal = SIZE_LONG
            elif donchian_short_breakout and htf_1d_bearish:
                desired_signal = -SIZE_SHORT
            # CRSI pullback entry in trend
            elif crsi_oversold and htf_1d_bullish and donchian_bullish:
                desired_signal = SIZE_LONG
            elif crsi_overbought and htf_1d_bearish and donchian_bearish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use Donchian direction with CRSI filter
            if donchian_bullish and crsi_bullish and htf_1d_bullish:
                desired_signal = SIZE_LONG
            elif donchian_bearish and crsi_bearish and htf_1d_bearish:
                desired_signal = -SIZE_SHORT
            # CRSI extreme entries
            elif crsi_oversold and not htf_1d_bearish:
                desired_signal = SIZE_LONG
            elif crsi_overbought and not htf_1d_bullish:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        # CRITICAL: Don't exit on every minor signal change, reduces fee churn
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if Donchian still bullish OR CRSI not extremely overbought
                if donchian_bullish and crsi_12h[i] < 80:
                    desired_signal = SIZE_LONG
                elif htf_1d_bullish and crsi_12h[i] < 70:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if Donchian still bearish OR CRSI not extremely oversold
                if donchian_bearish and crsi_12h[i] > 20:
                    desired_signal = -SIZE_SHORT
                elif htf_1d_bearish and crsi_12h[i] > 30:
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