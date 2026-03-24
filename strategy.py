#!/usr/bin/env python3
"""
Experiment #1495: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion within Trend

Hypothesis: After 1100+ failed strategies, the pattern is clear for lower TF:
1. 1h timeframe needs VERY strict filters to avoid fee drag (>100 trades/yr = fail)
2. Connors RSI (CRSI) has 75% win rate in academic literature for mean reversion
3. Must use HTF (4h) for DIRECTION, 1h only for ENTRY TIMING
4. Session filter (8-20 UTC) reduces trades by ~40% while keeping quality entries
5. LOOSE entry thresholds (CRSI<20/>80 vs <10/>90) to ensure trades happen

Key innovations:
- Connors RSI = (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
- 4h HMA(21) for trend bias (only long when price>4h_HMA, only short when price<4h_HMA)
- 1d HMA(21) for macro filter (avoid counter-macro trades)
- Session filter: only trade 8-20 UTC (London/NY overlap = best liquidity)
- Volume filter: LOOSE (0.8x avg) to ensure trades happen
- Position size: 0.20 (smaller for 1h to reduce fee impact)
- ATR stoploss: 2.5x trailing

Timeframe: 1h
HTF: 4h + 1d (call get_htf_data ONCE before loop!)
Target: 40-80 trades/train, Sharpe > 0.618 (beat current best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h1d_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, rsi_period) + RSI(streak, streak_period) + PercentRank(rank_period)) / 3
    
    Academic research shows 75% win rate for CRSI<10 long / CRSI>90 short
    We use looser thresholds (20/80) to ensure trades happen on 1h TF
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # Component 1: RSI of close (very short period for sensitivity)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive series for RSI calculation
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_positive[max(0, i-streak_period):i+1])
        avg_loss = np.mean(streak_negative[max(0, i-streak_period):i+1])
        if avg_loss > 1e-10:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        else:
            streak_rsi[i] = 100.0
    
    # Component 3: Percentile Rank of close over lookback
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        if np.any(np.isnan(window)):
            continue
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine components
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_close) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_close[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

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

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
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
    
    # Calculate and align HTF HMAs for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Smaller size for 1h to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after indicators are ready (CRSI needs 100+)
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === MACRO TREND (1d HMA) - avoid counter-macro trades ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - direction bias ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION (LOOSE) ===
        vol_confirmed = volume[i] > 0.8 * vol_sma[i] if vol_sma[i] > 0 else True
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds for trades) ===
        crsi_oversold = crsi[i] < 20.0  # Long entry (was <10, too strict)
        crsi_overbought = crsi[i] > 80.0  # Short entry (was >90, too strict)
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === DESIRED SIGNAL - MEAN REVERSION WITHIN TREND ===
        desired_signal = 0.0
        
        # LONG: Daily/4h bull + CRSI oversold + session + volume
        # Only long when HTF trend is bullish (mean reversion within uptrend)
        if daily_bull and inter_bull:
            # Strong long: extreme CRSI + session + volume
            if crsi_extreme_oversold and in_session and vol_confirmed:
                desired_signal = BASE_SIZE
            # Medium long: CRSI oversold + session (no volume req)
            elif crsi_oversold and in_session:
                desired_signal = BASE_SIZE * 0.7
            # Weak long: CRSI oversold only (no session, no volume)
            elif crsi_oversold:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT: Daily/4h bear + CRSI overbought + session + volume
        # Only short when HTF trend is bearish (mean reversion within downtrend)
        elif daily_bear and inter_bear:
            # Strong short: extreme CRSI + session + volume
            if crsi_extreme_overbought and in_session and vol_confirmed:
                desired_signal = -BASE_SIZE
            # Medium short: CRSI overbought + session (no volume req)
            elif crsi_overbought and in_session:
                desired_signal = -BASE_SIZE * 0.7
            # Weak short: CRSI overbought only (no session, no volume)
            elif crsi_overbought:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.5
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
                # Flip position
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