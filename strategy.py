#!/usr/bin/env python3
"""
Experiment #1593: 5m Primary + 15m/4h HTF — Session-Filtered Momentum with CRSI

Hypothesis: 5m timeframe is unexplored territory. Using HTF (4h/15m) for trend bias
and session filters (NY/London overlap 13-17 UTC) should reduce noise while allowing
precise 5m entries. Connors RSI (CRSI) provides faster reversal signals than standard RSI.

Key innovations:
1. SESSION FILTER: Only trade 13-17 UTC (NY session peak liquidity) - reduces false signals
2. 4h HMA for primary trend bias (never trade counter-trend on 5m)
3. 15m RSI for momentum confirmation (aligns with HTF direction)
4. CRSI(3,2,100) for precise 5m entry timing (faster than RSI14)
5. Volume spike confirmation (1.5x avg) to filter false breakouts
6. Small position size (0.15-0.20) due to higher trade frequency

Entry logic:
- LONG: 4h_HMA bullish + 15m_RSI>50 + CRSI<20 + volume>1.5x + session active
- SHORT: 4h_HMA bearish + 15m_RSI<50 + CRSI>80 + volume>1.5x + session active

Target: Sharpe>0.6, trades>=50 train, trades>=5 test, DD>-30%
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller due to more trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_crsi_momentum_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for faster reversal signals
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    streak = np.zeros(n, dtype=np.int64)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_vals = np.abs(streak[i-streak_period+1:i+1])
        if len(streak_vals) > 0:
            avg_streak = np.mean(streak_vals)
            streak_rsi[i] = 100 / (1 + avg_streak) if avg_streak > 0 else 50
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < close[i])
            percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def is_session_active(open_time_unix, start_hour=13, end_hour=17):
    """
    Check if timestamp is within trading session (UTC)
    Default: 13-17 UTC (NY session peak)
    """
    # Convert unix ms to datetime
    timestamp_ms = open_time_unix
    timestamp_s = timestamp_ms / 1000.0
    
    # Get hour UTC
    import datetime
    dt = datetime.datetime.utcfromtimestamp(timestamp_s)
    hour = dt.hour
    
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_32100 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    rsi_5m = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi_32100[i]) or np.isnan(rsi_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (13-17 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=13, end_hour=17)
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === MOMENTUM CONFIRMATION (15m RSI) ===
        rsi_15m = rsi_15m_aligned[i]
        rsi_15m_bullish = rsi_15m > 50
        rsi_15m_bearish = rsi_15m < 50
        
        # === CRSI EXTREMES (5m entry timing) ===
        crsi = crsi_32100[i]
        crsi_oversold = crsi < 25
        crsi_overbought = crsi > 75
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.5 if not np.isnan(vol_ratio[i]) else False
        
        # === ENTRY LOGIC (must generate trades - not too strict) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m RSI bullish + CRSI oversold + volume
        if price_above_4h and rsi_15m_bullish and crsi_oversold:
            if vol_confirmed and session_active:
                desired_signal = SIZE_STRONG
            elif session_active:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m RSI bearish + CRSI overbought + volume
        elif price_below_4h and rsi_15m_bearish and crsi_overbought:
            if vol_confirmed and session_active:
                desired_signal = -SIZE_STRONG
            elif session_active:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals