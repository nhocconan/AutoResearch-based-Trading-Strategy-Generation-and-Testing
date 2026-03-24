#!/usr/bin/env python3
"""
Experiment #301: 15m Primary + 1h/4h/1d HTF — Session RSI Pullback with HTF Trend Filter v1

Hypothesis: 15m timeframe needs VERY selective entries to avoid fee drag. Using 4h HMA for
major trend direction, 1h RSI for pullback timing, and UTC session filter (00-12) for
London/NY overlap liquidity. Connors RSI extremes catch oversold bounces in uptrends.

Key design choices:
1. 4h HMA(21) = major trend filter (only trade in trend direction)
2. 1h RSI(7) pullback = entry timing (RSI<35 in uptrend, RSI>65 in downtrend)
3. Session filter = UTC 00-12 only (reduces trades by ~50%, avoids Asian session chop)
4. CRSI(3,2,100) = confirms extreme oversold/overbought conditions
5. Volume spike = 1.5x 20-bar average (confirms institutional interest)
6. Position size: 0.15-0.25 (smaller for 15m frequency, discrete levels)
7. Stoploss: 2.0x ATR from entry (tighter for lower TF)

Target: 40-80 trades/year, Sharpe>0.40, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_htf_trend_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < pr_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Streak RSI
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        up_count = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        streak_rsi[i] = 100.0 * up_count / streak_period
    
    # Percent Rank of returns
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        if len(window) > 0 and not np.isnan(returns[i]):
            count_below = np.sum(window < returns[i])
            percent_rank[i] = 100.0 * count_below / len(window)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * average"""
    n = len(volume)
    if n < period:
        return np.full(n, False)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_avg)
    
    return spike

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = pd.to_datetime(open_time, unit='ms').hour
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h RSI aligned to 15m
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=7)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, 200)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_1h_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 only) ===
        hour = get_session_hour(open_time[i])
        in_session = (hour >= 0 and hour < 12)
        
        # === 4h HTF TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 1d MAJOR TREND FILTER ===
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === 1h RSI PULLBACK ===
        rsi_1h = rsi_1h_aligned[i]
        rsi_oversold = not np.isnan(rsi_1h) and rsi_1h < 35.0
        rsi_overbought = not np.isnan(rsi_1h) and rsi_1h > 65.0
        
        # === CRSI EXTREMES ===
        crsi_extreme_low = False
        crsi_extreme_high = False
        if not np.isnan(crsi[i]):
            crsi_extreme_low = crsi[i] < 20.0
            crsi_extreme_high = crsi[i] > 80.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 1h RSI oversold + CRSI extreme + session + volume
        if htf_4h_bull and htf_1d_bull and rsi_oversold and in_session:
            if crsi_extreme_low and above_sma200:
                # Strong signal: all confluence
                if vol_confirmed:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif above_sma200:
                # Weaker signal: no CRSI extreme
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 1h RSI overbought + CRSI extreme + session + volume
        elif htf_4h_bear and htf_1d_bear and rsi_overbought and in_session:
            if crsi_extreme_high and below_sma200:
                # Strong signal: all confluence
                if vol_confirmed:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif below_sma200:
                # Weaker signal: no CRSI extreme
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals