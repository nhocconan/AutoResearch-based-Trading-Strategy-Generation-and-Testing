#!/usr/bin/env python3
"""
Experiment #658: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: 30m timeframe with strict HTF filters can capture intraday moves while 
avoiding fee drag. Connors RSI (CRSI) has documented 75% win rate for mean reversion.
Combined with 4h HMA trend filter and 1d Choppiness regime, this should work in 
both bull and bear markets.

Key innovations:
1. Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 — proven mean reversion signal
2. 4h HMA(21) for trend direction — only trade with HTF trend
3. 1d Choppiness Index — avoid trading in extreme chop (CHOP > 65)
4. Volume filter — only enter when volume > 0.8x 20-bar average
5. Session filter — only trade 8-20 UTC (highest liquidity)
6. Conservative sizing (0.20) for lower TF to survive whipsaws

Why 30m can work:
- HTF (4h/1d) determines DIRECTION, 30m only for ENTRY TIMING
- This gives HTF trade frequency (~40-60/year) with 30m execution precision
- CRSI extremes (<10 or >90) are rare enough to avoid fee drag
- Volume + session filters eliminate low-quality signals

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_hma_chop_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — proven mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Long signal: CRSI < 10 (oversold)
    Short signal: CRSI > 90 (overbought)
    
    Documented win rate: 75%+ for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3) of close
    def calc_rsi(price, period):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
        
        # Pad to match length
        rsi = np.concatenate([[np.nan] * period, rsi[period:]])
        if len(rsi) < n:
            rsi = np.pad(rsi, (0, n - len(rsi)), constant_values=np.nan)
        return rsi
    
    rsi_3 = calc_rsi(close, rsi_period)
    
    # RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute values for RSI calculation
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    
    # RSI(2) of streak
    streak_gain = np.where(streak_abs > np.roll(streak_abs, 1), streak_abs - np.roll(streak_abs, 1), 0)
    streak_loss = np.where(streak_abs < np.roll(streak_abs, 1), np.roll(streak_abs, 1) - streak_abs, 0)
    streak_gain[0] = 0
    streak_loss[0] = 0
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    
    rsi_streak = np.concatenate([[np.nan] * streak_period, rsi_streak[streak_period:]])
    if len(rsi_streak) < n:
        rsi_streak = np.pad(rsi_streak, (0, n - len(rsi_streak)), constant_values=np.nan)
    
    # PercentRank(100) — where does current return rank vs last 100 bars?
    returns = np.diff(close) / (close[:-1] + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i-1] if i > 0 else 0
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100
    
    # Calculate CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_3 + rsi_streak + percent_rank) / 3
    
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 65 = too choppy (no trade), < 55 = tradeable
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
    
    # Calculate 30m indicators (primary timeframe)
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.22
    SIZE_SHORT = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]):
            continue
        if np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(chop_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        vol_ok = vol_ratio > 0.7  # At least 70% of average volume
        
        # === REGIME FILTER (1d Choppiness) ===
        chop_val = chop_1d_aligned[i]
        too_choppy = chop_val > 65.0  # Don't trade in extreme chop
        tradeable_regime = chop_val < 60.0  # OK to trade
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === CRSI SIGNALS (Connors RSI extremes) ===
        crsi_oversold = crsi_30m[i] < 12.0  # Long entry
        crsi_overbought = crsi_30m[i] > 88.0  # Short entry
        
        # === ENTRY CONDITIONS (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold + HTF bullish + in session + volume OK + not too choppy
        if crsi_oversold and htf_4h_bullish and in_session and vol_ok and tradeable_regime:
            desired_signal = SIZE_LONG
        
        # SHORT: CRSI overbought + HTF bearish + in session + volume OK + not too choppy
        elif crsi_overbought and htf_4h_bearish and in_session and vol_ok and tradeable_regime:
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
                # Hold long if HTF still bullish AND CRSI not extremely overbought
                if htf_4h_bullish and crsi_30m[i] < 80.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish AND CRSI not extremely oversold
                if htf_4h_bearish and crsi_30m[i] > 20.0:
                    desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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