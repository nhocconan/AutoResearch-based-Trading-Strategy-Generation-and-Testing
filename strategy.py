#!/usr/bin/env python3
"""
Experiment #395: 1h Primary + 4h/1d HTF — Regime-Adaptive CRSI with Session/Volume Filter

Hypothesis: 1h timeframe can work IF we use HTF for direction and relax entry thresholds.
Key learnings from failed 1h strategies (#385, #390):
- Too many filters = 0 trades (the #1 killer)
- Need RELAXED CRSI thresholds (<30/>70 not <10/>90)
- Session filter (8-20 UTC) reduces Asian session whipsaws
- Volume filter ensures real moves, not fake breakouts

Strategy Design:
1. 1d HMA = ultra-HTF bias (only trade with this trend)
2. 4h HMA = primary trend direction (confirm before entry)
3. Choppiness Index = regime detector (chop vs trend)
4. Connors RSI = entry timing (relaxed <30/>70 for 1h)
5. Session filter = only 8-20 UTC (avoid Asian session)
6. Volume filter = >0.8x 20-bar average
7. Position size = 0.25 (smaller for 1h, target 40-70 trades/year)
8. Stoploss = 2.0x ATR trailing

Target: Sharpe > 0.612, 40-70 trades/year, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_session_4h1d_relaxed_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    diff = 2.0 * wma_half - wma_full
    hma = diff.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Relaxed thresholds for 1h: <30 oversold, >70 overbought
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, period=rsi_period)
    
    # RSI of Streak - consecutive up/down bars
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= streak_period:
            streak_rsi[i] = 100.0
        elif streak[i] <= -streak_period:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0 + 25.0 * streak[i]
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank - percentile of today's return vs last pr_period bars
    returns = close_s.pct_change()
    percent_rank = np.full(n, 50.0)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i]
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[i] > window).sum() / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chopperness = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chopperness = np.nan_to_num(chopperness, nan=50.0)
    chopperness = np.clip(chopperness, 0, 100)
    
    return chopperness

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000
    utc_hour = pd.to_datetime(ts_seconds, unit='s').hour
    return utc_hour

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
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for bias (4h and 1d)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (target 40-70 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === HTF BIAS (1d HMA) - ultra HTF direction ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) - confirm direction ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Relaxed for 1h
        is_trending = chop[i] < 45.0  # Relaxed for 1h
        
        # === CRSI EXTREMES (relaxed for 1h) ===
        crsi_oversold = crsi[i] < 30.0  # Relaxed from <10
        crsi_overbought = crsi[i] > 70.0  # Relaxed from >90
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - need HTF bias + regime + CRSI + session + volume
        long_bias = price_above_hma_1d and price_above_hma_4h
        
        if long_bias and in_session and volume_ok:
            if is_trending:
                # Trend regime: enter on pullback (CRSI oversold)
                if crsi_oversold:
                    desired_signal = BASE_SIZE
            elif is_choppy:
                # Range regime: enter at lower bound (CRSI oversold)
                if crsi_oversold:
                    desired_signal = BASE_SIZE
            else:
                # Neutral regime: need strong CRSI signal
                if crsi[i] < 25.0:
                    desired_signal = BASE_SIZE
        
        # SHORT SETUP - need HTF bias + regime + CRSI + session + volume
        short_bias = price_below_hma_1d and price_below_hma_4h
        
        if short_bias and in_session and volume_ok:
            if is_trending:
                # Trend regime: enter on rally (CRSI overbought)
                if crsi_overbought:
                    desired_signal = -BASE_SIZE
            elif is_choppy:
                # Range regime: enter at upper bound (CRSI overbought)
                if crsi_overbought:
                    desired_signal = -BASE_SIZE
            else:
                # Neutral regime: need strong CRSI signal
                if crsi[i] > 75.0:
                    desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 65:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 35:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_4h:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_4h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and long_bias:
                desired_signal = BASE_SIZE
            elif position_side < 0 and short_bias:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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