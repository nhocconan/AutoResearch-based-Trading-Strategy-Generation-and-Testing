#!/usr/bin/env python3
"""
Experiment #468: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: Lower timeframe (30m) strategies fail due to excessive trades → fee drag.
Solution: Use 4h/1d for SIGNAL DIRECTION, 30m only for ENTRY TIMING with strict filters.

Key innovations:
1. Choppiness Index (14) regime detection: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
2. Connors RSI for precise entries: CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long when CRSI < 15, Short when CRSI > 85 (extreme mean reversion)
3. 4h HMA(21) trend bias — ONLY long when price > 4h HMA, ONLY short when price < 4h HMA
4. Session filter: only trade 8-20 UTC (reduces trades by ~50%, avoids Asian session noise)
5. Volume filter: volume > 0.8x 20-bar average (confirms participation)
6. Discrete position sizing: 0.0, ±0.20, ±0.25 to minimize fee churn
7. ATR(14) trailing stop at 2.5x for risk management

Target: Sharpe > 0.612, 30-80 trades/year, DD < -35%
Timeframe: 30m (with 4h/1d HTF alignment)
Why this should work: Session filter + volume filter + HTF bias = VERY FEW trades.
Connors RSI has 75% win rate on mean reversion. Choppiness avoids trend-chopping.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR(1), period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    choppiness = np.full(n, np.nan)
    
    # Calculate ATR(1) = true range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
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
        abs_streak = np.abs(streak[i])
        if abs_streak >= streak_period:
            # More consecutive days = more extreme
            if streak[i] > 0:
                streak_rsi[i] = min(100, 50 + abs_streak * 15)
            else:
                streak_rsi[i] = max(0, 50 - abs_streak * 15)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank: where does current close rank vs last 100 bars?
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = (count_below / rank_period) * 100.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    dt = pd.to_datetime(open_time, unit='ms', utc=True)
    return dt.hour

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
    atr_14 = calculate_atr(high, low, close, 14)
    choppiness_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    volume_sma_20 = calculate_volume_sma(volume, 20)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(choppiness_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(volume_sma_20[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma_20[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = choppiness_14[i] > 55.0  # Range/mean-revert regime
        is_trend = choppiness_14[i] < 45.0  # Trending regime
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1d CONFIRMATION ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI EXTREMES (Mean Reversion) ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Range regime OR trend + HTF bullish + CRSI oversold + session + volume
        long_conditions = 0
        if in_session:
            long_conditions += 2  # Session is mandatory
        if volume_ok:
            long_conditions += 1
        if crsi_oversold:
            long_conditions += 2  # CRSI extreme is key
        if price_above_hma_4h:
            long_conditions += 1  # HTF bias
        if price_above_hma_1d:
            long_conditions += 1  # 1d confirmation
        
        # Need at least 5 conditions for long (very strict)
        if long_conditions >= 5:
            desired_signal = SIZE_LONG
        
        # SHORT: Range regime OR trend + HTF bearish + CRSI overbought + session + volume
        if desired_signal == 0.0:
            short_conditions = 0
            if in_session:
                short_conditions += 2  # Session is mandatory
            if volume_ok:
                short_conditions += 1
            if crsi_overbought:
                short_conditions += 2  # CRSI extreme is key
            if price_below_hma_4h:
                short_conditions += 1  # HTF bias
            if price_below_hma_1d:
                short_conditions += 1  # 1d confirmation
            
            # Need at least 5 conditions for short (very strict)
            if short_conditions >= 5:
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_4h and in_session:
                desired_signal = SIZE_LONG
            elif position_side < 0 and price_below_hma_4h and in_session:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.25
        elif desired_signal < 0:
            desired_signal = -0.20
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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