#!/usr/bin/env python3
"""
Experiment #630: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness Regime + Volume Session

Hypothesis: Lower timeframe (1h) can work IF we use extremely strict confluence filters
to limit trades to 30-60/year. Previous 1h attempts failed due to too many trades → fee drag.

Key improvements over failed 1h strategies (#490, #595, #598):
1. Connors RSI (CRSI) for precise mean-reversion entries — proven 75% win rate in research
2. Choppiness Index regime filter — only mean-revert when CHOP > 55 (range market)
3. 4h HMA for trend bias — only long when 4h bullish, only short when 4h bearish
4. Volume filter — only trade when volume > 0.8x 20-bar average (avoid low-liquidity traps)
5. Session filter — only 8-20 UTC (high liquidity, avoid Asia overnight whipsaw)
6. VERY strict CRSI thresholds (< 8 long, > 92 short) to minimize false signals
7. Conservative sizing (0.20) to survive 2022-style crashes

Why this should work where other 1h failed:
- CRSI combines 3 signals (RSI(3), RSI-Streak, PercentRank) for more reliable extremes
- CHOP regime prevents trend-following logic in chop (where it gets whipsawed)
- 4h HMA ensures we trade WITH higher-timeframe momentum, not against it
- Session + volume filters cut 60%+ of potential trades, reducing fee drag

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_hma4h_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — Larry Connors' mean-reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Very short-term RSI for quick oversold/overbought detection
    2. RSI-Streak(2): RSI of consecutive up/down days (streak strength)
    3. PercentRank(100): Where current price ranks vs last 100 bars (0-100%)
    
    Entry: CRSI < 10 (extreme oversold) or CRSI > 90 (extreme overbought)
    We use even stricter: < 8 for long, > 92 for short to reduce false signals.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    rsi_3 = np.clip(rsi_3, 0, 100)
    
    # Component 2: RSI of Streak (consecutive up/down bars)
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
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank(100) — where current close ranks in last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i]) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine into CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: >55 = chop (mean revert), <45 = trend (trend follow)
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE = 0.20  # Conservative sizing for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1h[i] > 55.0
        is_trending = chop_1h[i] < 45.0
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 12h for stronger confirmation
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        # Extreme oversold (long entry)
        crsi_oversold = crsi_1h[i] < 8.0
        # Extreme overbought (short entry)
        crsi_overbought = crsi_1h[i] > 92.0
        
        # Moderate levels for exit/hold
        crsi_neutral_low = crsi_1h[i] < 30.0
        crsi_neutral_high = crsi_1h[i] > 70.0
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: CRSI extreme oversold + 4h not strongly bearish + session + volume
            if crsi_oversold and not htf_4h_bearish and in_session and volume_ok:
                desired_signal = SIZE
            # Short: CRSI extreme overbought + 4h not strongly bullish + session + volume
            elif crsi_overbought and not htf_4h_bullish and in_session and volume_ok:
                desired_signal = -SIZE
        
        # === REGIME 2: TRENDING MARKET (Trend Pullback) ===
        elif is_trending:
            # Long: 4h bullish + CRSI pullback (not overbought) + session + volume
            if htf_4h_bullish and crsi_neutral_low and in_session and volume_ok:
                desired_signal = SIZE
            # Short: 4h bearish + CRSI pullback (not oversold) + session + volume
            elif htf_4h_bearish and crsi_neutral_high and in_session and volume_ok:
                desired_signal = -SIZE
        
        # === REGIME 3: NEUTRAL (Wait for extreme CRSI only) ===
        else:
            # Only trade extreme CRSI with strong HTF confirmation
            if crsi_oversold and htf_12h_bullish and in_session and volume_ok:
                desired_signal = SIZE
            elif crsi_overbought and htf_12h_bearish and in_session and volume_ok:
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
        
        # === HOLD LOGIC — Maintain position if conditions still favorable ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish OR CRSI not yet overbought
                if htf_4h_bullish and crsi_1h[i] < 75.0:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish OR CRSI not yet oversold
                if htf_4h_bearish and crsi_1h[i] > 25.0:
                    desired_signal = -SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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