#!/usr/bin/env python3
"""
Experiment #065: 1h Primary + 4h/1d HTF — Volatility Regime + CRSI + Session Filter

Hypothesis: Previous CRSI strategies failed because they lacked proper regime filtering
and traded during low-liquidity periods. This version adds:
1. Bollinger Band Width percentile for volatility regime (squeeze vs expansion)
2. 200 SMA for long-term trend bias (only trade with the trend)
3. 4h HMA alignment for HTF confirmation
4. Session filter (8-20 UTC) to avoid low-liquidity whipsaws
5. Volume confirmation (>1.2x average)
6. Looser CRSI thresholds (15/85 not 10/90) for trade generation

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1h (target 30-60 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_volregime_crsi_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    def wma(data, span):
        res = np.full(len(data), np.nan)
        w = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            res[i] = np.sum(data[i - span + 1:i + 1] * w) / np.sum(w)
        return res
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    double_wma = 2.0 * wma_half - wma_full
    hma = wma(double_wma, sqrt_p)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """RSI Streak component for Connors RSI"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_rsi = calculate_rsi(streak, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component for Connors RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan)
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + streak_rsi + pr) / 3.0
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, sma, lower

def calculate_bb_width_percentile(close, lookback=100):
    """Bollinger Band Width percentile for volatility regime"""
    n = len(close)
    if n < lookback + 20:
        return np.full(n, np.nan)
    
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    bb_width_pct = np.full(n, np.nan)
    for i in range(lookback, n):
        if np.isnan(bb_width[i]):
            continue
        window = bb_width[i-lookback+1:i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) < lookback // 2:
            continue
        rank = np.sum(valid_window[:-1] <= bb_width[i])
        bb_width_pct[i] = 100.0 * rank / (len(valid_window) - 1)
    
    return bb_width_pct

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    """Volume SMA for confirmation filter"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000.0
    hour = int((ts_seconds % 86400) / 3600)
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
    
    # Calculate and align 4h/1d HMA for HTF trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    bb_width_pct = calculate_bb_width_percentile(close, lookback=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative size for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
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
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_utc_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLATILITY REGIME (BB Width Percentile) ===
        low_vol_regime = bb_width_pct[i] < 30.0  # Volatility squeeze (expecting expansion)
        high_vol_regime = bb_width_pct[i] > 70.0  # Volatility expansion (mean reversion likely)
        
        # === LONG-TERM TREND BIAS (200 SMA) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === HTF TREND ALIGNMENT (4h and 1d HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment (at least 4h agrees with 1d)
        htf_bull_aligned = hma_4h_bull and hma_1d_bull
        htf_bear_aligned = hma_4h_bear and hma_1d_bear
        
        # === CRSI EXTREMES (Looser than failed attempts) ===
        crsi_oversold = crsi[i] < 20.0  # Was 10, too strict
        crsi_overbought = crsi[i] > 80.0  # Was 90, too strict
        
        # === VOLUME CONFIRMATION ===
        vol_above_avg = volume[i] > 1.2 * vol_sma[i]
        
        # === BOLLINGER POSITION ===
        near_lower_bb = close[i] < bb_lower[i] * 1.005
        near_upper_bb = close[i] > bb_upper[i] * 0.995
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Multiple confluence required
        long_conditions = 0
        if in_session:
            long_conditions += 1
        if above_sma200:
            long_conditions += 1
        if htf_bull_aligned:
            long_conditions += 1
        if crsi_oversold:
            long_conditions += 1
        if near_lower_bb or low_vol_regime:
            long_conditions += 1
        if vol_above_avg:
            long_conditions += 1
        
        # Need at least 4 out of 6 conditions for long
        if long_conditions >= 4 and crsi_oversold and above_sma200:
            desired_signal = SIZE
        
        # SHORT ENTRY: Multiple confluence required
        short_conditions = 0
        if in_session:
            short_conditions += 1
        if below_sma200:
            short_conditions += 1
        if htf_bear_aligned:
            short_conditions += 1
        if crsi_overbought:
            short_conditions += 1
        if near_upper_bb or high_vol_regime:
            short_conditions += 1
        if vol_above_avg:
            short_conditions += 1
        
        # Need at least 4 out of 6 conditions for short
        if short_conditions >= 4 and crsi_overbought and below_sma200:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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