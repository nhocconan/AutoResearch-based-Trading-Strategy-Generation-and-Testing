#!/usr/bin/env python3
"""
Experiment #595: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Volume Session

Hypothesis: Connors RSI (CRSI) provides superior mean-reversion signals vs standard RSI,
especially in bear/range markets (2022 crash, 2025 test period). Combined with:
1. Choppiness Index regime filter (CHOP>55=range/mean-revert, CHOP<45=trend/follow)
2. 1d HMA for secular trend bias (only long if price>1d HMA, only short if price<1d HMA)
3. 4h KAMA for intermediate trend confirmation
4. Volume filter (>0.8x 20-bar avg) to avoid low-liquidity entries
5. Session filter (8-20 UTC) to trade during high-volume hours only
6. ATR trailing stop (2.5x) for risk management

Why Connors RSI over standard RSI:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- More sensitive to short-term extremes, 75% win rate in research
- Better at catching reversals in bear markets vs standard RSI(14)

Why 1h timeframe:
- More entry opportunities than 4h/12h but fewer than 15m/30m
- Can use 4h/1d for direction, 1h for precise entry timing
- Target: 30-60 trades/year (strict confluence prevents overtrading)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_vol_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - 3-component mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term RSI for quick reversal signals
    2. RSI_Streak(2): RSI of streak (consecutive up/down days)
    3. PercentRank(100): Where current return ranks vs last 100 bars
    
    Entry: CRSI < 10 (oversold) for long, CRSI > 90 (overbought) for short
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
    rsi_short = np.clip(rsi_short, 0, 100)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank of returns
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100
    
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = returns[i-pr_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine components
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + rsi_streak + percent_rank) / 3
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = price_change / sum_changes
        else:
            er[i] = 0.0
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

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

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Volume moving average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    kama_4h_raw = calculate_kama(df_4h['close'].values, period=10, fast_period=2, slow_period=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.20
    
    # Position tracking
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
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(kama_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] >= 0.8 * vol_avg_20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1h[i] > 55.0
        is_trending = chop_1h[i] < 45.0
        
        # === HTF TREND BIAS ===
        htf_4h_bullish = close[i] > kama_4h_aligned[i]
        htf_4h_bearish = close[i] < kama_4h_aligned[i]
        
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 15
        crsi_overbought = crsi_1h[i] > 85
        
        # Extreme CRSI for stronger signals
        crsi_deep_oversold = crsi_1h[i] < 10
        crsi_deep_overbought = crsi_1h[i] > 90
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: Deep CRSI oversold + HTF 1d not bearish + session + volume
            if crsi_deep_oversold and not htf_1d_bearish and in_session and volume_ok:
                desired_signal = SIZE_LONG
            # Short: Deep CRSI overbought + HTF 1d not bullish + session + volume
            elif crsi_deep_overbought and not htf_1d_bullish and in_session and volume_ok:
                desired_signal = -SIZE_SHORT
            # Moderate CRSI with strong HTF confirmation
            elif crsi_oversold and htf_4h_bullish and htf_1d_bullish and in_session and volume_ok:
                desired_signal = SIZE_LONG
            elif crsi_overbought and htf_4h_bearish and htf_1d_bearish and in_session and volume_ok:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Following with CRSI pullback) ===
        elif is_trending:
            # Long: HTF bullish + CRSI pullback (not overbought) + session + volume
            if htf_4h_bullish and htf_1d_bullish and crsi_1h[i] < 50 and in_session and volume_ok:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + CRSI pullback (not oversold) + session + volume
            elif htf_4h_bearish and htf_1d_bearish and crsi_1h[i] > 50 and in_session and volume_ok:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL (Wait for extreme CRSI) ===
        else:
            # Only enter on extreme CRSI with HTF confirmation
            if crsi_deep_oversold and htf_4h_bullish and in_session and volume_ok:
                desired_signal = SIZE_LONG
            elif crsi_deep_overbought and htf_4h_bearish and in_session and volume_ok:
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
                # Hold long if HTF 4h still bullish
                if htf_4h_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF 4h still bearish
                if htf_4h_bearish:
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