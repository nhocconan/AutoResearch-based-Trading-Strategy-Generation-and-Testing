#!/usr/bin/env python3
"""
Experiment #708: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness + Session + Volume

Hypothesis: Lower timeframe (30m) can work IF we use HTF for direction and 30m only
for entry timing. Key insight from failures: too many filters = 0 trades (#699,#700,#703,#705,#707).

This strategy combines:
1. 4h HMA(21) for trend bias (direction filter)
2. Connors RSI(3,2,100) for mean-reversion entry timing
3. Choppiness Index(14) for regime detection (range vs trend)
4. Session filter (8-20 UTC) for liquidity
5. Volume filter (>0.8x 20-bar avg) for confirmation

Why this should work:
- CRSI is proven mean-reversion signal (75% win rate in ranges)
- 4h trend filter prevents counter-trend trades that failed in #698
- Session filter avoids low-liquidity whipsaw (Asian session)
- Volume confirmation ensures real moves, not noise
- 30m TF gives more entries than 4h but fewer than 15m (target 40-80 trades/year)

CRITICAL: Loosen thresholds to ensure trades generate (learned from 0-trade failures)
- CRSI < 25 / > 75 (not 10/90 which is too rare)
- CHOP > 50 (not 61.8 which is too strict)
- Volume > 0.7x avg (not 1.2x which filters too much)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_session_volume_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(close,3) + RSI(streak,2) + PercentRank(close,100)) / 3
    
    Streak RSI: consecutive up/down days, then RSI of that streak series
    PercentRank: where current close ranks vs last 100 closes (0-100)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period:
        return crsi
    
    # RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak (use absolute values for RSI calc)
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_avg_gain = np.concatenate([[np.nan], streak_avg_gain])
    streak_avg_loss = np.concatenate([[np.nan], streak_avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # PercentRank(100) - where current close ranks in last 100
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        rank = np.sum(window < close[i]) / pr_period * 100
        percent_rank[i] = rank
    
    # Combine
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # ATR sum and ATR(period)
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    hh_ll = hh - ll
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * (atr_sum / (atr_period + 1e-10)) / (hh_ll + 1e-10) * np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(series, period):
    """Hull Moving Average."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
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
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_30m = calculate_choppiness_index(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    
    # Volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for lower TF to reduce fee impact
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 100 for CRSI + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(chop_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            continue
        if atr_30m[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour
        ts = pd.to_datetime(open_time[i], unit='ms', utc=True)
        hour_utc = ts.hour
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_sma[i]
        
        # === TREND BIAS (HTF HMA) ===
        # 4h trend direction
        trend_bullish_4h = close[i] > hma_4h_aligned[i]
        trend_bearish_4h = close[i] < hma_4h_aligned[i]
        
        # 1d trend confirmation (optional but helps)
        trend_bullish_1d = close[i] > hma_1d_aligned[i]
        trend_bearish_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both agree
        trend_strong_bullish = trend_bullish_4h and trend_bullish_1d
        trend_strong_bearish = trend_bearish_4h and trend_bearish_1d
        
        # === CHOPPINESS REGIME ===
        is_range = chop_30m[i] > 50  # Range/choppy market
        is_trending = chop_30m[i] < 45  # Trending market
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_30m[i] < 25
        crsi_overbought = crsi_30m[i] > 75
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # Must have: bullish HTF trend + CRSI oversold + (range OR trending with confirmation) + session + volume
        if trend_bullish_4h and crsi_oversold and in_session and volume_ok:
            if is_range:
                # Mean reversion in range within uptrend
                desired_signal = current_size
            elif is_trending and trend_strong_bullish:
                # Pullback entry in strong trend
                desired_signal = current_size
            elif not is_trending:
                # Neutral regime - still take signal with 4h confirmation
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY ===
        if trend_bearish_4h and crsi_overbought and in_session and volume_ok:
            if is_range:
                # Mean reversion in range within downtrend
                desired_signal = -current_size
            elif is_trending and trend_strong_bearish:
                # Pullback entry in strong downtrend
                desired_signal = -current_size
            elif not is_trending:
                # Neutral regime
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish and CRSI not overbought
                if trend_bullish_4h and crsi_30m[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and CRSI not oversold
                if trend_bearish_4h and crsi_30m[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: CRSI overbought OR 4h trend reverses
        if in_position and position_side > 0:
            if crsi_30m[i] > 85:
                desired_signal = 0.0
            elif close[i] < hma_4h_aligned[i]:
                desired_signal = 0.0
        
        # Short exit: CRSI oversold OR 4h trend reverses
        if in_position and position_side < 0:
            if crsi_30m[i] < 15:
                desired_signal = 0.0
            elif close[i] > hma_4h_aligned[i]:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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