#!/usr/bin/env python3
"""
Experiment #728: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: Lower timeframe (30m) strategies fail due to excessive trades and fee drag.
This strategy uses:
1. 4h Choppiness Index for regime detection (range vs trend)
2. 1d HMA for ultra-long-term trend bias
3. Connors RSI (CRSI) on 30m for precise mean-reversion entries
4. Session filter (8-20 UTC) to avoid low-liquidity whipsaws
5. Volume filter (>0.8x avg) to confirm participation
6. Very strict confluence: need 3+ conditions to trigger

Key innovations:
- CRSI combines RSI(3) + Streak RSI(2) + PercentRank(100) for superior mean-reversion signals
- CHOP regime filter prevents trend-following in ranges and vice versa
- Session filter reduces false signals during Asian/late US sessions
- Small position size (0.20-0.25) for lower TF risk management
- Target: 40-80 trades/year (strict enough to avoid fee drag)

Timeframe: 30m (use 4h/1d for direction, 30m for entry timing)
Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies ranging vs trending markets.
    CHOP > 61.8 = range (mean reversion favored)
    CHOP < 38.2 = trend (trend following favored)
    Formula: 100 * LOG10(SUM(ATR(1), period) / (Highest High(period) - Lowest Low(period))) / LOG10(period)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - superior mean-reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close_change, 100)) / 3
    
    RSI(3): Fast RSI for short-term momentum
    RSI_Streak(2): RSI of consecutive up/down streaks
    PercentRank(100): Percentile rank of current price change over last 100 periods
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_direction = np.sign(streak)
    
    # RSI of streak (using absolute streak values, direction handled separately)
    streak_delta = np.diff(streak_abs)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
    avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # PercentRank(100) - percentile rank of price change
    price_change = np.diff(close)
    price_change = np.concatenate([[0], price_change])
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = price_change[i-rank_period+1:i+1]
        current = price_change[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_fast) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_fast[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time_arr):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = ((open_time_arr // 1000) % 86400) // 3600
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
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    sma_200_30m = calculate_sma(close, period=200)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF indicators
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract session hours
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22  # Smaller size for lower TF
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(350, n):  # Need buffer for all indicators + HTF alignment + CRSI rank_period
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200_30m[i]) or np.isnan(vol_sma_30m[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (> 0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_sma_30m[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        chop_value = chop_4h_aligned[i]
        is_range = chop_value > 55  # Range market (mean reversion favored)
        is_trend = chop_value < 45  # Trending market (trend follow favored)
        
        # === TREND BIAS (1d HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY CONDITIONS (need 3+ confluence) ===
        long_confluence = 0
        
        # Confluence 1: CRSI extremely oversold (< 15)
        if crsi_30m[i] < 15:
            long_confluence += 1
        
        # Confluence 2: Range regime OR bullish 1d trend
        if is_range or trend_1d_bullish:
            long_confluence += 1
        
        # Confluence 3: Above SMA200 (for mean reversion in uptrend)
        if above_sma200:
            long_confluence += 1
        
        # Confluence 4: In trading session
        if in_session:
            long_confluence += 1
        
        # Confluence 5: Volume confirmation
        if volume_ok:
            long_confluence += 1
        
        # Confluence 6: Not in strong downtrend (1d HMA not strongly bearish)
        if not (trend_1d_bearish and chop_value < 40):
            long_confluence += 1
        
        # Need 4+ confluence for long entry (very strict)
        if long_confluence >= 4 and crsi_30m[i] < 20:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS (need 3+ confluence) ===
        short_confluence = 0
        
        # Confluence 1: CRSI extremely overbought (> 85)
        if crsi_30m[i] > 85:
            short_confluence += 1
        
        # Confluence 2: Range regime OR bearish 1d trend
        if is_range or trend_1d_bearish:
            short_confluence += 1
        
        # Confluence 3: Below SMA200 (for mean reversion in downtrend)
        if below_sma200:
            short_confluence += 1
        
        # Confluence 4: In trading session
        if in_session:
            short_confluence += 1
        
        # Confluence 5: Volume confirmation
        if volume_ok:
            short_confluence += 1
        
        # Confluence 6: Not in strong uptrend (1d HMA not strongly bullish)
        if not (trend_1d_bullish and chop_value < 40):
            short_confluence += 1
        
        # Need 4+ confluence for short entry (very strict)
        if short_confluence >= 4 and crsi_30m[i] > 80:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals possible, use 1d HMA as tiebreaker
        if desired_signal > 0 and crsi_30m[i] > 80:
            if trend_1d_bearish:
                desired_signal = -current_size
            else:
                desired_signal = 0.0
        
        if desired_signal < 0 and crsi_30m[i] < 20:
            if trend_1d_bullish:
                desired_signal = current_size
            else:
                desired_signal = 0.0
        
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
                # Hold long if CRSI not extremely overbought and trend intact
                if crsi_30m[i] < 75 and (trend_1d_bullish or is_range):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not extremely oversold and trend intact
                if crsi_30m[i] > 25 and (trend_1d_bearish or is_range):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI extremely overbought or trend strongly reverses
            if crsi_30m[i] > 85 or (trend_1d_bearish and chop_value < 40):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI extremely oversold or trend strongly reverses
            if crsi_30m[i] < 15 or (trend_1d_bullish and chop_value < 40):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
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
                # Position flip
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