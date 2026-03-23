#!/usr/bin/env python3
"""
Experiment #938: 30m Primary + 4h/1d HTF — Connors RSI + HTF Trend + Session Filter

Hypothesis: For 30m timeframe, use HTF (4h/1d) for SIGNAL DIRECTION and 30m only for 
ENTRY TIMING. This gives HTF trade frequency (30-80/year) with 30m execution precision.

Key components:
1. 1d HMA(21) — macro trend bias (only trade in direction of macro trend)
2. 4h HMA(21) — medium-term confirmation (must align with 1d)
3. Connors RSI(3,2,100) — proven mean reversion entry (CRSI<15 long, >85 short)
4. Volume filter — volume > 0.8x 20-period average (confirms participation)
5. Session filter — only 8-20 UTC (higher liquidity, lower slippage)
6. ATR(14) trailing stop — 2.5x ATR exit

Why this should work:
- CRSI has 75% win rate in research (Larry Connors)
- HTF trend filter prevents counter-trend trades (major failure cause)
- Session filter avoids low-liquidity periods (reduces slippage)
- 30m entries within 4h/1d trend = fewer trades, higher quality

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive Sharpe
Timeframe: 30m (use 4h/1d for direction, 30m for timing)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_htf_trend_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    streak_gain = np.where(streak > 0, abs_streak, 0)
    streak_loss = np.where(streak < 0, abs_streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    avg_streak_gain = np.concatenate([[np.nan] * streak_period, avg_streak_gain[streak_period:]])
    avg_streak_loss = np.concatenate([[np.nan] * streak_period, avg_streak_loss[streak_period:]])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100 - (100 / (1 + streak_rs))
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_lower = np.sum(window[:-1] < current)
        percent_rank[i] = 100 * count_lower / (rank_period - 1)
    
    # Combine CRSI
    for i in range(n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / average volume."""
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            ratio[i] = volume[i] / avg_vol
    
    return ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    timestamp_s = open_time / 1000
    utc_hour = pd.to_datetime(timestamp_s, unit='s').hour
    return utc_hour

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
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_ratio_30m = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(vol_ratio_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_30m[i] < 15
        crsi_overbought = crsi_30m[i] > 85
        crsi_extreme_oversold = crsi_30m[i] < 10
        crsi_extreme_overbought = crsi_30m[i] > 90
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio_30m[i] > 0.8
        
        # === SESSION FILTER ===
        session_ok = in_session
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Macro bull + 4h bull + CRSI oversold + volume + session
        if macro_bull and trend_4h_bullish and crsi_oversold and volume_ok and session_ok:
            desired_signal = BASE_SIZE
        # Secondary: Macro bull + CRSI extreme oversold (relaxed session/volume)
        elif macro_bull and crsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        # Tertiary: Both HTF trends bullish + CRSI oversold (strong confluence)
        elif macro_bull and trend_4h_bullish and crsi_30m[i] < 25:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY ===
        # Primary: Macro bear + 4h bear + CRSI overbought + volume + session
        if macro_bear and trend_4h_bearish and crsi_overbought and volume_ok and session_ok:
            desired_signal = -BASE_SIZE
        # Secondary: Macro bear + CRSI extreme overbought (relaxed session/volume)
        elif macro_bear and crsi_extreme_overbought:
            desired_signal = -REDUCED_SIZE
        # Tertiary: Both HTF trends bearish + CRSI overbought (strong confluence)
        elif macro_bear and trend_4h_bearish and crsi_30m[i] > 75:
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
                # Hold long if macro trend intact and CRSI not overbought
                if macro_bull and crsi_30m[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro trend intact and CRSI not oversold
                if macro_bear and crsi_30m[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro trend reverses
            if macro_bear and trend_4h_bearish:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought (take profit)
            if crsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro trend reverses
            if macro_bull and trend_4h_bullish:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold (take profit)
            if crsi_extreme_oversold:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
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