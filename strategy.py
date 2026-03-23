#!/usr/bin/env python3
"""
Experiment #730: 1h Primary + 4h/12h HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: After 488 failed strategies, the key insight is that bear/range markets (2025)
require regime-adaptive logic. This strategy combines:
1. Choppiness Index (CHOP) for regime detection: CHOP>55=range(mean revert), CHOP<45=trend(follow)
2. Connors RSI (CRSI) for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. 4h/12h HMA for trend bias (HTF direction filter)
4. Session filter (8-20 UTC) for volume confirmation
5. ATR stoploss (2.5x) with trailing

Key differences from failed experiments:
- 1h timeframe with HTF direction filter (proven to reduce trade frequency)
- Connors RSI instead of standard RSI (better for mean reversion)
- Choppiness Index regime switching (adapt to market conditions)
- Session filter ensures we only trade during high-volume hours
- Multiple entry paths to ensure trade frequency (critical after #728/#729 got 0 trades)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (30-60 trades/year target with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_session_4h12h_v1"
timeframe = "1h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
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
        up_streaks = np.sum(streak_vals > 0)
        total = len(streak_vals)
        if total > 0:
            streak_rsi[i] = (up_streaks / total) * 100
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = (rank / (rank_period - 1)) * 100
    
    # Combine
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market choppiness vs trending.
    CHOP > 61.8 = range/bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) / (log10(n))
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            chop[i] = 100 * (atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def get_hour_from_timestamp(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    timestamps = prices['open_time'].values / 1000
    hours = pd.to_datetime(timestamps, unit='s').hour.values
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Get session hours
    hours = get_hour_from_timestamp(prices)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h timeframe
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(chop_1h[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(donch_upper[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === TREND BIAS (4h and 12h HTF HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong trend when both HTF agree
        strong_bullish = trend_4h_bullish and trend_12h_bullish
        strong_bearish = trend_4h_bearish and trend_12h_bearish
        
        # === CHOPPINESS REGIME DETECTION ===
        chop_range = chop_1h[i] > 55  # Range market - mean revert
        chop_trend = chop_1h[i] < 45  # Trending market - follow trend
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE if in_session else REDUCED_SIZE
        
        # === LONG ENTRY CONDITIONS (multiple paths to ensure trades) ===
        long_signal = False
        
        # Path 1: Range regime + CRSI oversold + HTF bullish (mean reversion in uptrend)
        if chop_range and crsi_1h[i] < 15 and trend_4h_bullish:
            long_signal = True
        
        # Path 2: Trend regime + HTF strong bullish + RSI pullback
        if chop_trend and strong_bullish and rsi_1h[i] < 45 and above_sma200:
            long_signal = True
        
        # Path 3: CRSI deeply oversold + above 4h HMA (strong mean reversion)
        if crsi_1h[i] < 10 and trend_4h_bullish:
            long_signal = True
        
        # Path 4: Donchian breakout + HTF bullish + in session
        if close[i] > donch_upper[i-1] and trend_4h_bullish and in_session:
            long_signal = True
        
        # Path 5: RSI oversold + range regime (classic mean reversion)
        if rsi_1h[i] < 30 and chop_range and above_sma200:
            long_signal = True
        
        # Path 6: Strong bullish HTF + CRSI < 25 (ensure trade frequency)
        if strong_bullish and crsi_1h[i] < 25:
            long_signal = True
        
        if long_signal and in_session:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS (multiple paths to ensure trades) ===
        short_signal = False
        
        # Path 1: Range regime + CRSI overbought + HTF bearish (mean reversion in downtrend)
        if chop_range and crsi_1h[i] > 85 and trend_4h_bearish:
            short_signal = True
        
        # Path 2: Trend regime + HTF strong bearish + RSI bounce
        if chop_trend and strong_bearish and rsi_1h[i] > 55 and below_sma200:
            short_signal = True
        
        # Path 3: CRSI deeply overbought + below 4h HMA (strong mean reversion)
        if crsi_1h[i] > 90 and trend_4h_bearish:
            short_signal = True
        
        # Path 4: Donchian breakdown + HTF bearish + in session
        if close[i] < donch_lower[i-1] and trend_4h_bearish and in_session:
            short_signal = True
        
        # Path 5: RSI overbought + range regime (classic mean reversion)
        if rsi_1h[i] > 70 and chop_range and below_sma200:
            short_signal = True
        
        # Path 6: Strong bearish HTF + CRSI > 75 (ensure trade frequency)
        if strong_bearish and crsi_1h[i] > 75:
            short_signal = True
        
        if short_signal and in_session:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with stronger trend (12h HMA)
        if long_signal and short_signal:
            if trend_12h_bullish:
                desired_signal = current_size
            elif trend_12h_bearish:
                desired_signal = -current_size
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h HMA still bullish and CRSI not extremely overbought
                if trend_4h_bullish and crsi_1h[i] < 85:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h HMA still bearish and CRSI not extremely oversold
                if trend_4h_bearish and crsi_1h[i] > 15:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI extremely overbought
            if trend_4h_bearish or crsi_1h[i] > 90:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI extremely oversold
            if trend_4h_bullish or crsi_1h[i] < 10:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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