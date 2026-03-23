#!/usr/bin/env python3
"""
Experiment #128: 30m Primary + 4h/1d HTF — Regime-Adaptive Pullback Strategy

Hypothesis: Lower TF strategies fail due to excessive trades (>200/yr) causing fee drag.
This strategy uses PROVEN pattern from rules: Choppiness Index regime + Connors RSI + HTF HMA.

Key innovations:
1) 4h HMA(21) for TREND DIRECTION — only trade pullbacks in HTF trend direction
2) 1d Choppiness Index for REGIME — CHOP>55=range(mean revert), CHOP<45=trend(follow)
3) 30m Connors RSI for ENTRY TIMING — CRSI<15 long, CRSI>85 short (extreme pullbacks)
4) Session filter — only trade 8-20 UTC (highest volume, lowest manipulation)
5) Volume confirmation — volume > 1.2x 20-bar avg (filters false signals)
6) Strict position sizing — 0.20 base, 0.30 max with confluence (lower TF = smaller size)

Why this should work on 30m:
- HTF (4h/1d) determines direction = fewer whipsaws
- 30m only for entry timing = precision without overtrading
- Session + volume filters = ~40-80 trades/year (acceptable fee drag)
- CRSI proven 75% win rate on pullbacks
- Regime adaptation = works in both trending and ranging markets

Position size: 0.20 base, 0.30 max (smaller for lower TF per rules)
Stoploss: 2.5*ATR(14) trailing
Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_hma_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)  # avoid div by zero
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = np.maximum(streak_s.diff(), 0)
    streak_loss = -np.minimum(streak_s.diff(), 0)
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

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
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 4h HMA slope
    hma_4h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-1]) and hma_4h_aligned[i-1] != 0:
            hma_4h_slope[i] = (hma_4h_aligned[i] - hma_4h_aligned[i-1]) / hma_4h_aligned[i-1] * 100
        else:
            hma_4h_slope[i] = 0.0
    
    # Calculate 1d Choppiness Index for regime
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    chop_1d = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    hma_30m_21 = calculate_hma(close, period=21)
    hma_30m_50 = calculate_hma(close, period=50)
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Session filter: only trade 8-20 UTC
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    in_session = (utc_hours >= 8) & (utc_hours <= 20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(hma_30m_21[i]) or np.isnan(hma_30m_50[i]) or np.isnan(crsi_30m[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        hma_4h_slope_positive = hma_4h_slope[i] > 0.3
        hma_4h_slope_negative = hma_4h_slope[i] < -0.3
        
        # === REGIME FILTER (1d CHOP) ===
        chop_value = chop_1d_aligned[i]
        is_trending = chop_value < 45.0
        is_ranging = chop_value > 55.0
        
        # === 30m TREND FILTER ===
        hma_30m_bullish = hma_30m_21[i] > hma_30m_50[i]
        hma_30m_bearish = hma_30m_21[i] < hma_30m_50[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_value = crsi_30m[i]
        crsi_oversold = crsi_value < 15.0
        crsi_overbought = crsi_value > 85.0
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.2
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 4h trend up + (trending regime + CRSI pullback OR ranging regime + CRSI extreme)
        # + volume + session
        if price_above_hma_4h:
            # Trending regime: follow pullback to CRSI<20
            if is_trending and crsi_oversold and hma_30m_bullish:
                if volume_confirmed and session_ok:
                    new_signal = POSITION_SIZE_BASE
                    if hma_4h_slope_positive and volume_ratio > 1.5:
                        new_signal = POSITION_SIZE_MAX
            # Ranging regime: mean revert at CRSI<15
            elif is_ranging and crsi_value < 15.0:
                if volume_confirmed and session_ok:
                    new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # Require: 4h trend down + (trending regime + CRSI rally OR ranging regime + CRSI extreme)
        # + volume + session
        if price_below_hma_4h:
            # Trending regime: follow rally to CRSI>80
            if is_trending and crsi_overbought and hma_30m_bearish:
                if volume_confirmed and session_ok:
                    new_signal = -POSITION_SIZE_BASE
                    if hma_4h_slope_negative and volume_ratio > 1.5:
                        new_signal = -POSITION_SIZE_MAX
            # Ranging regime: mean revert at CRSI>85
            elif is_ranging and crsi_value > 85.0:
                if volume_confirmed and session_ok:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if regime and HTF trend still intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h trend intact and not overbought
                if price_above_hma_4h and crsi_value < 80.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h trend intact and not oversold
                if price_below_hma_4h and crsi_value > 20.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_4h_slope_negative:
                new_signal = 0.0
            # Exit on CRSI overbought (take profit)
            if crsi_value > 85.0:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope_positive:
                new_signal = 0.0
            # Exit on CRSI oversold (take profit)
            if crsi_value < 15.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals