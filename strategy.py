#!/usr/bin/env python3
"""
Experiment #1248: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: Previous 30m strategies failed with 0 trades (Sharpe=0.000) due to overly strict
filters. Key changes to ensure trades while maintaining quality:
1. Connors RSI (CRSI) for entry timing - proven 75% win rate in mean reversion
2. 4h HMA(21) for trend DIRECTION (not mandatory, just biases entries)
3. 1d HMA(50) for macro regime filter
4. Session filter (8-20 UTC) to reduce trades and avoid Asian session noise
5. Volume filter (loose: >0.7x avg) to avoid dead periods
6. LOOSENED thresholds: CRSI <25/>75 (not <20/>80) to ensure trade generation
7. Position size: 0.25 (conservative for 30m TF)
8. Stoploss: 2.5x ATR trailing

Target: Sharpe > 0.612, trades 40-80/year (160-320 over 4yr train), >=12 test
Timeframe: 30m (use 4h/1d for direction, 30m for entry timing only)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_4h1d_hma_session_vol_atr_v2"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) - very short term momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_rsi = pd.Series(np.concatenate([[0], gain])).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_rsi = pd.Series(np.concatenate([[0], loss])).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    mask = loss_rsi[1:] > 1e-10
    rsi_short[1:][mask] = 100.0 - (100.0 / (1.0 + gain_rsi[1:][mask] / loss_rsi[1:][mask]))
    rsi_short[~mask] = 50.0
    
    # RSI Streak(2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        positive = np.sum(streak_vals > 0)
        streak_rsi[i] = 100.0 * positive / streak_period if streak_period > 0 else 50.0
    
    # PercentRank(100) - where current price ranks in last 100 bars
    pr = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < window[-1])
        pr[i] = 100.0 * rank / (rank_period - 1) if rank_period > 1 else 50.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Rolling average volume"""
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend filters
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Local HMA for additional trend confirmation
    hma_local = calculate_hma(close, period=34)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 30m TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or atr[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_local[i]) or np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (loose to ensure trades) ===
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - BIAS, NOT MANDATORY ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND (30m HMA) ===
        trend_local_bull = close[i] > hma_local[i]
        trend_local_bear = close[i] < hma_local[i]
        
        # === CONNORS RSI EXTREMES (LOOSENED for trade generation) ===
        crsi_oversold = crsi[i] < 25.0  # Was <20, loosened
        crsi_overbought = crsi[i] > 75.0  # Was >80, loosened
        
        # === ENTRY CONDITIONS (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: Macro bull OR 4h bull + CRSI oversold + session + volume
        # At least 3 of: macro_bull, trend_4h_bull, trend_local_bull, crsi_oversold, in_session, volume_ok
        long_score = 0
        if macro_bull: long_score += 1
        if trend_4h_bull: long_score += 1
        if trend_local_bull: long_score += 1
        if crsi_oversold: long_score += 2  # CRSI is primary trigger
        if in_session: long_score += 1
        if volume_ok: long_score += 1
        
        if long_score >= 4 and crsi_oversold:  # CRSI mandatory + 3 other confluence
            desired_signal = BASE_SIZE
        
        # SHORT: Macro bear OR 4h bear + CRSI overbought + session + volume
        short_score = 0
        if macro_bear: short_score += 1
        if trend_4h_bear: short_score += 1
        if trend_local_bear: short_score += 1
        if crsi_overbought: short_score += 2  # CRSI is primary trigger
        if in_session: short_score += 1
        if volume_ok: short_score += 1
        
        if short_score >= 4 and crsi_overbought:  # CRSI mandatory + 3 other confluence
            desired_signal = -BASE_SIZE
        
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
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals