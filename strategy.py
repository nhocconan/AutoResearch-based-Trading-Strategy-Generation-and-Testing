#!/usr/bin/env python3
"""
Experiment #698: 30m Primary + 4h/1d HTF — Choppiness Regime + CRSI + Session Filter

Hypothesis: Lower timeframe (30m) strategies fail due to excessive trade frequency.
Solution: Use 4h/1d HMA for TREND DIRECTION, 30m only for ENTRY TIMING.
Add strict confluence filters: Choppiness regime + Connors RSI + Session (8-20 UTC)
+ Volume filter. Target 30-80 trades/year to minimize fee drag.

Key innovations vs failed #688:
1. CRSI (Connors RSI) instead of simple RSI - better mean reversion signal
2. Choppiness Index for regime detection - only mean revert in ranges
3. Session filter (8-20 UTC) - avoid low liquidity hours
4. Volume confirmation - only trade when volume > 0.8x average
5. HTF trend bias mandatory - never trade against 4h+1d HMA direction
6. Discrete signal sizes (0.20, 0.30) to reduce churn

Why this should work:
- 30m entries within 4h trend = HTF trade frequency with LTF precision
- CRSI < 10 / > 90 are rare events = fewer trades naturally
- Session filter cuts ~40% of potential trades (overnight hours)
- Volume filter eliminates low-liquidity false breakouts
- Multiple HTF filters (4h + 1d HMA) prevent counter-trend disasters

Target: Sharpe > 0.612, trades >= 120 train (30/yr), >= 38 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_volume_htf_hma_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
    
    # Rolling sum of ATR and highest high - lowest low
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return np.clip(chop, 0, 100)

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long opportunity)
    CRSI > 90 = overbought (short opportunity)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
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
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
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
    
    # Percent Rank (100) - where does current return rank vs last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            percentile = np.sum(returns < current_return) / len(returns) * 100
            percent_rank[i] = percentile
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
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
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_avg(volume, period=20):
    """Rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)."""
    # Convert ms to seconds, then to datetime
    timestamps = pd.to_datetime(open_time_array, unit='ms', utc=True)
    hours = timestamps.hour.values
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
    
    # Calculate primary (30m) indicators
    chop_30m = calculate_choppiness_index(high, low, close, period=14)
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    sma_200_30m = calculate_sma(close, period=200)
    vol_avg_30m = calculate_volume_avg(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment + CRSI rank
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if np.isnan(chop_30m[i]) or np.isnan(sma_200_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Avoid overnight hours when liquidity is low and spreads are wide
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        # Only trade when volume is at least 80% of average
        volume_ok = volume[i] >= 0.8 * vol_avg_30m[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (favor mean reversion)
        # CHOP < 45 = trending (favor trend following)
        is_range_regime = chop_30m[i] > 55
        is_trend_regime = chop_30m[i] < 45
        
        # === TREND BIAS (HTF HMA) ===
        # 4h HMA for intermediate trend, 1d HMA for major trend
        trend_bullish_4h = close[i] > hma_4h_aligned[i]
        trend_bearish_4h = close[i] < hma_4h_aligned[i]
        trend_bullish_1d = close[i] > hma_1d_aligned[i]
        trend_bearish_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTF agree
        trend_strong_bullish = trend_bullish_4h and trend_bullish_1d
        trend_strong_bearish = trend_bearish_4h and trend_bearish_1d
        trend_neutral = not trend_strong_bullish and not trend_strong_bearish
        
        # === SMA200 FILTER (long-term trend on 30m) ===
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # Reduce size in high volatility
        atr_ratio = atr_30m[i] / (np.nanmedian(atr_30m[max(0,i-100):i+1]) + 1e-10)
        if atr_ratio > 1.5:
            current_size = REDUCED_SIZE
        
        # === MEAN REVERSION MODE (Range regime + CRSI extremes) ===
        # Only enter mean reversion trades in range regime, WITH HTF trend bias
        if is_range_regime and in_session and volume_ok:
            # Long: CRSI oversold + HTF bullish or neutral + above SMA200
            if crsi_30m[i] < 15 and (trend_bullish_4h or trend_neutral) and above_sma200:
                desired_signal = current_size
            
            # Short: CRSI overbought + HTF bearish or neutral + below SMA200
            elif crsi_30m[i] > 85 and (trend_bearish_4h or trend_neutral) and below_sma200:
                desired_signal = -current_size
            
            # Weaker signals (extreme CRSI but less confluence)
            elif crsi_30m[i] < 10 and trend_bullish_4h:
                desired_signal = REDUCED_SIZE
            elif crsi_30m[i] > 90 and trend_bearish_4h:
                desired_signal = -REDUCED_SIZE
        
        # === TREND FOLLOWING MODE (Trend regime + pullback entry) ===
        # In trend regime, wait for pullback (CRSI moderate) then enter with trend
        elif is_trend_regime and in_session and volume_ok:
            # Long pullback: strong bullish trend + CRSI dipped but not extreme
            if trend_strong_bullish and above_sma200 and 30 <= crsi_30m[i] <= 50:
                desired_signal = current_size
            
            # Short pullback: strong bearish trend + CRSI rallied but not extreme
            elif trend_strong_bearish and below_sma200 and 50 <= crsi_30m[i] <= 70:
                desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and 4h trend intact
                if crsi_30m[i] < 80 and trend_bullish_4h:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and 4h trend intact
                if crsi_30m[i] > 20 and trend_bearish_4h:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: CRSI overbought OR trend reverses below both HTF HMA
        if in_position and position_side > 0:
            if crsi_30m[i] > 85:
                desired_signal = 0.0
            elif close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]:
                desired_signal = 0.0
        
        # Short exit: CRSI oversold OR trend reverses above both HTF HMA
        if in_position and position_side < 0:
            if crsi_30m[i] < 15:
                desired_signal = 0.0
            elif close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]:
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