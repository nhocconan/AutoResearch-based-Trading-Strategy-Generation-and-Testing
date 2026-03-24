#!/usr/bin/env python3
"""
Experiment #1628: 30m Primary + 4h/1d HTF — Regime-Adaptive with Session Filter

Hypothesis: 30m strategies fail due to TOO MANY trades (>200/yr) causing fee drag.
Solution: Use 4h/1d for SIGNAL DIRECTION, 30m only for ENTRY TIMING.
Apply 4 confluence filters: HTF trend + CHOP regime + CRSI extremes + session(8-20 UTC).

Key innovations:
1. Dual HTF bias (4h HMA + 1d HMA must agree) - reduces false signals
2. CHOP regime detection - trend follow when CHOP<45, mean revert when CHOP>55
3. Connors RSI (CRSI) - more responsive than standard RSI for entry timing
4. Session filter - only trade 8-20 UTC (London/NY overlap, avoid Asian noise)
5. Volume confirmation - volume > 0.8x 20-bar average
6. Discrete sizing (0.0, ±0.25, ±0.30) - minimize fee churn
7. ATR trailing stop 2.5x - protect capital

Timeframe: 30m (required)
HTF: 4h + 1d (use mtf_data helper - call ONCE before loop)
Target: 30-80 trades/year, Sharpe > 0.618, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_crsi_4h1d_hma_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    
    RSI(Streak): RSI of consecutive up/down days
    PercentRank: percentage of prior returns lower than current return
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values (treat absolute streak as "price")
    streak_abs = np.abs(streak)
    streak_abs[streak_abs == 0] = 0.001  # avoid zero
    rsi_streak = calculate_rsi(streak_abs, streak_period)
    
    # Percent Rank - rolling percentile of returns
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current_return = returns[i]
            if np.isnan(current_return):
                percent_rank[i] = 50.0
            else:
                percentile = np.sum(valid < current_return) / len(valid)
                percent_rank[i] = percentile * 100.0
    
    # Combine CRSI
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_close[mask] + rsi_streak[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_session_hour(open_time_ms):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time_ms // 3600000) % 24

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
    
    # Calculate and align HTF HMAs for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for lower TF
    MAX_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_ma[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = choppy/range, CHOP < 45 = trending
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === HTF TREND BIAS (4h + 1d must agree) ===
        bull_4h = close[i] > hma_4h_aligned[i]
        bear_4h = close[i] < hma_4h_aligned[i]
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        
        # Both HTFs must agree for strong bias
        strong_bull = bull_4h and bull_1d
        strong_bear = bear_4h and bear_1d
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 25 = oversold (long opportunity)
        # CRSI > 75 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === PRIMARY SIGNAL LOGIC ===
        desired_signal = 0.0
        
        # Only trade during active session with volume confirmation
        if in_session and volume_confirmed:
            # REGIME 1: TRENDING MARKET - Follow HTF trend on CRSI pullback
            if is_trending:
                # Long: Strong bull bias + CRSI pullback from overbought
                if strong_bull and crsi_oversold:
                    desired_signal = BASE_SIZE
                # Short: Strong bear bias + CRSI bounce from oversold
                elif strong_bear and crsi_overbought:
                    desired_signal = -BASE_SIZE
            
            # REGIME 2: CHOPPY MARKET - Mean reversion with HTF filter
            elif is_choppy:
                # Long: HTF neutral/bull + CRSI extremely oversold
                if (bull_4h or not bear_4h) and crsi[i] < 20.0:
                    desired_signal = BASE_SIZE
                # Short: HTF neutral/bear + CRSI extremely overbought
                elif (bear_4h or not bull_4h) and crsi[i] > 80.0:
                    desired_signal = -BASE_SIZE
        
        # Hold existing position if no new signal
        if in_position and desired_signal == 0.0:
            desired_signal = BASE_SIZE if position_side > 0 else -BASE_SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = MAX_SIZE if desired_signal > BASE_SIZE else BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -MAX_SIZE if desired_signal < -BASE_SIZE else -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals