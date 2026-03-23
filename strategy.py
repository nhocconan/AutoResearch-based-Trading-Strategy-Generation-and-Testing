#!/usr/bin/env python3
"""
Experiment #710: 1h Primary + 4h/12h HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: Lower timeframe (1h) strategies fail due to fee drag from too many trades.
Solution: Use 4h/12h HTF for TREND DIRECTION, 1h only for ENTRY TIMING within HTF trend.
Add Choppiness Index regime filter + Connors RSI (proven 75% win rate) + session filter (8-20 UTC).

Key improvements:
1. CHOP(14) regime: >55 = range (mean revert), <45 = trend (follow HTF)
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — more responsive than standard RSI
3. 4h HMA(21) for strong trend bias (proven in best strategies)
4. Session filter: only trade 8-20 UTC (high volume, less noise)
5. Volume filter: only enter if volume > 0.7x 20-bar avg
6. Conservative sizing: 0.25 (lower TF = smaller size to survive fee drag)
7. Looser CRSI thresholds (15/85 not 10/90) to ensure trade frequency

Why this should work:
- 1h TF with HTF filter = HTF trade frequency with 1h execution precision
- CRSI proven in literature for mean reversion with 75% win rate
- Choppiness Index prevents trend strategies in range markets (#1 cause of failures)
- Session filter reduces noise trades (failed #698/#708 had no session filter)
- Looser thresholds avoid 0-trade failures (#699/#700/#703/#705/#707)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_session_volume_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100 - (100 / (streak_abs[i] + 1))
        else:
            streak_rsi[i] = 100 / (streak_abs[i] + 1)
    
    # Percent Rank - where current return ranks vs last 100 bars
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and returns[-1] != np.nan:
            pct_rank[i] = 100 * np.sum(returns[:-1] < returns[-1]) / (len(returns) - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging.
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period * 2:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    hh_ll = hh - ll
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop_raw, 0, 100)
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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time_col):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time_col // (1000 * 60 * 60)) % 24
    return hours

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
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_200_1h = calculate_sma(close, period=200)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h TF
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Extract hours for session filter
    hours = get_hour_from_open_time(open_time)
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment + CRSI rank
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_200_1h[i]) or np.isnan(vol_avg_20[i]):
            continue
        if atr_1h[i] <= 1e-10 or vol_avg_20[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_range = chop_1h[i] > 55  # Range market
        chop_trend = chop_1h[i] < 45  # Trending market
        # Neutral zone: 45-55 (no trades or reduced size)
        
        # === TREND BIAS (4h + 12h HMA confluence) ===
        # Both HTF must agree for strong signal
        trend_bullish_4h = close[i] > hma_4h_aligned[i]
        trend_bearish_4h = close[i] < hma_4h_aligned[i]
        trend_bullish_12h = close[i] > hma_12h_aligned[i]
        trend_bearish_12h = close[i] < hma_12h_aligned[i]
        
        # Strong trend: both 4h and 12h agree
        trend_strong_bull = trend_bullish_4h and trend_bullish_12h
        trend_strong_bear = trend_bearish_4h and trend_bearish_12h
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === TRENDING REGIME (CHOP < 45) ===
        if chop_trend and in_session and volume_ok:
            # Long: Strong HTF bullish + CRSI pullback + above SMA200
            if trend_strong_bull and crsi_1h[i] < 35 and above_sma200:
                desired_signal = current_size
            
            # Short: Strong HTF bearish + CRSI bounce + below SMA200
            elif trend_strong_bear and crsi_1h[i] > 65 and below_sma200:
                desired_signal = -current_size
            
            # Weaker signals (single HTF confirmation)
            elif trend_bullish_4h and crsi_1h[i] < 25:
                desired_signal = REDUCED_SIZE
            elif trend_bearish_4h and crsi_1h[i] > 75:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (CHOP > 55) ===
        elif chop_range and in_session and volume_ok:
            # Mean reversion: CRSI extremes only (no HTF bias needed in range)
            if crsi_1h[i] < 20:  # Oversold in range
                desired_signal = REDUCED_SIZE
            elif crsi_1h[i] > 80:  # Overbought in range
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Only take extreme CRSI signals with HTF confirmation
            if crsi_1h[i] < 15 and trend_strong_bull:
                desired_signal = REDUCED_SIZE
            elif crsi_1h[i] > 85 and trend_strong_bear:
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
                # Hold long if CRSI not overbought and HTF trend intact
                if crsi_1h[i] < 75 and trend_bullish_4h:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and HTF trend intact
                if crsi_1h[i] > 25 and trend_bearish_4h:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if crsi_1h[i] > 80:  # CRSI overbought exit
                desired_signal = 0.0
            elif close[i] < hma_4h_aligned[i]:  # HTF trend broken
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if crsi_1h[i] < 20:  # CRSI oversold exit
                desired_signal = 0.0
            elif close[i] > hma_4h_aligned[i]:  # HTF trend broken
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
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
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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