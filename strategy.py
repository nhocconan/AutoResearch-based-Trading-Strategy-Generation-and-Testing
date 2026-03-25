#!/usr/bin/env python3
"""
Experiment #1453: 5m Primary + 15m/4h HTF — Session-Filtered Momentum with CRSI

Hypothesis: 5m timeframe can work with EXTREME selectivity using:
1. 4h HMA(21) for major trend bias (NEVER trade counter-trend)
2. 15m RSI(14) for momentum confirmation (aligns with 4h direction)
3. 5m Connors RSI (CRSI) for precise entry timing (mean reversion within trend)
4. Session filter (08-20 UTC) for London/NY overlap volume
5. Volume spike confirmation (1.5x 20-bar avg)

Why this should work on 5m:
- 4h trend filter prevents counter-trend disasters (main 5m failure mode)
- CRSI catches pullbacks within established trends (75% win rate literature)
- Session filter avoids low-volume whipsaw periods (00-08 UTC)
- Small size (0.15-0.20) accounts for higher fee drag on 5m
- Loose CRSI thresholds (20/80 not 10/90) guarantee sufficient trades

CRSI Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(3): Fast momentum
- RSI_Streak(2): Consecutive up/down days momentum
- PercentRank(100): Where current price sits in recent range

Entry logic (LOOSE enough for trades):
- LONG: 4h_HMA_bullish + 15m_RSI>45 + CRSI<25 + volume>1.5x + session_active
- SHORT: 4h_HMA_bearish + 15m_RSI<55 + CRSI>75 + volume>1.5x + session_active

Target: Sharpe>0.6, trades>=50/train, trades>=5/test, DD>-35%, 50-120 trades/year
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller due to fee drag on 5m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_crsi_session_hma4h_rsi15m_v1"
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
    Connors RSI (CRSI) - Composite mean reversion indicator
    Formula: (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Where current price sits in recent N-day range
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) on price
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Calculate streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak - convert streak to positive for RSI calc
    streak_positive = np.abs(streak)
    streak_direction = np.sign(streak)
    
    # Simple RSI on streak magnitude
    delta_streak = np.diff(streak_positive)
    gain_streak = np.where(delta_streak > 0, delta_streak, 0)
    loss_streak = np.where(delta_streak < 0, -delta_streak, 0)
    gain_streak = np.insert(gain_streak, 0, 0)
    loss_streak = np.insert(loss_streak, 0, 0)
    
    avg_gain_streak = pd.Series(gain_streak).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_loss_streak = pd.Series(loss_streak).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss_streak != 0
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_gain_streak[mask] / avg_loss_streak[mask]
    rsi_streak[mask] = 100 - (100 / (1 + rs_streak[mask]))
    
    # Apply direction: positive streak = high RSI, negative = low RSI
    rsi_streak = np.where(streak >= 0, rsi_streak, 100 - rsi_streak)
    
    # PercentRank(100): position of current close in recent range
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        window = close[i - rank_period + 1:i + 1]
        if len(window) == rank_period:
            count_below = np.sum(window[:-1] < close[i])
            percent_rank[i] = count_below / (rank_period - 1) * 100
    
    # CRSI = average of three components
    crsi = np.full(n, np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def is_session_active(open_time, start_hour=8, end_hour=20):
    """
    Check if timestamp is within active trading session (UTC)
    08-20 UTC captures London open through NY close overlap
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_15m = get_htf_data(prices, '15m')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for spike detection
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    # Warmup period (need CRSI rank_period + HTF alignment)
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
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
        
        # === SESSION FILTER (08-20 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === MOMENTUM CONFIRMATION (15m RSI) ===
        rsi_15m = rsi_15m_aligned[i]
        momentum_bullish = rsi_15m > 45  # Not oversold in uptrend
        momentum_bearish = rsi_15m < 55  # Not overbought in downtrend
        
        # === ENTRY SIGNAL (CRSI mean reversion within trend) ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 25  # Loose threshold for trades
        crsi_overbought = crsi_val > 75
        
        # === VOLUME SPIKE ===
        volume_spike = volume[i] > 1.5 * vol_sma[i]
        
        # === ENTRY LOGIC (LOOSE - must generate 50-120 trades/year) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m momentum + CRSI oversold + volume + session
        if price_above_4h and momentum_bullish and crsi_oversold:
            if volume_spike and session_active:
                desired_signal = SIZE_STRONG
            elif session_active:  # Allow without volume spike for more trades
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m momentum + CRSI overbought + volume + session
        elif price_below_4h and momentum_bearish and crsi_overbought:
            if volume_spike and session_active:
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