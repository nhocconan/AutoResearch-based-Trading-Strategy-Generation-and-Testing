#!/usr/bin/env python3
"""
Experiment #745: 1h Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + HTF HMA Trend

Hypothesis: After 498 failed strategies, clear patterns emerge for lower TF success:
1. 1h strategies fail due to TOO MANY trades → fee drag kills profit (#735, #738, #740)
2. Need 3+ CONFLUENCE filters: HTF trend + regime + entry timing + session/volume
3. Choppiness Index as META-FILTER: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
4. Connors RSI for precise pullback entries within HTF trend (proven 75% win rate)
5. Session filter (8-20 UTC) avoids low-volume whipsaw periods
6. Volume filter (>0.8x avg) confirms breakout validity

Strategy design:
1. 1d HMA(21) for macro trend bias (highest TF = strongest signal)
2. 4h HMA(21) for intermediate trend confirmation
3. Choppiness Index(14) regime filter: <45=trend, >55=range
4. Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 for entry timing
5. Volume filter: current > 0.8 * SMA20(volume)
6. Session filter: only trade 8-20 UTC (high liquidity periods)
7. ATR(14) trailing stop 2.5x for risk management
8. Discrete signals: 0.0, ±0.20, ±0.25 (smaller size for 1h TF)

Target: Sharpe > 0.612, trades 40-80/year, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year per rules)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_4h1d_session_vol_v1"
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
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of closes in lookback period that are below current close
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2)
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_changes = np.zeros(streak_period)
        for j in range(streak_period):
            streak_changes[j] = 1 if streak[i-j] > 0 else (0 if streak[i-j] == 0 else -1)
        
        gain_streak = np.sum(np.where(streak_changes > 0, streak_changes, 0))
        loss_streak = np.abs(np.sum(np.where(streak_changes < 0, streak_changes, 0)))
        
        if loss_streak > 1e-10:
            rs_streak = gain_streak / loss_streak
            streak_rsi[i] = 100 - (100 / (1 + rs_streak))
        else:
            streak_rsi[i] = 100
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i]
        count_below = np.sum(lookback < close[i])
        percent_rank[i] = 100 * count_below / rank_period
    
    # Combine into Connors RSI
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume SMA for filter
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
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
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(chop_1h[i]) or np.isnan(crsi_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma20[i]) or vol_sma20[i] <= 1e-10:
            continue
        
        # Extract hour from timestamp for session filter
        # open_time is in milliseconds
        hour_utc = (prices['open_time'].iloc[i] // 3600000) % 24
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma20[i]
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === TREND CONFIRMATION (4h HTF HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        trending_regime = chop_1h[i] < 45  # Trending market
        ranging_regime = chop_1h[i] > 55  # Range market
        
        # === CONNORS RSI ENTRY SIGNALS ===
        crsi_oversold = crsi_1h[i] < 20  # Strong pullback in uptrend
        crsi_overbought = crsi_1h[i] > 80  # Strong rally in downtrend
        crsi_neutral_long = 25 < crsi_1h[i] < 45  # Moderate pullback
        crsi_neutral_short = 55 < crsi_1h[i] < 75  # Moderate rally
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (3+ confluence required) ===
        long_signal = False
        
        # Path 1: Strong trend regime + 1d/4h bullish + CRSI oversold + volume + session
        if trending_regime and trend_1d_bullish and trend_4h_bullish and crsi_oversold and volume_ok and in_session:
            long_signal = True
        
        # Path 2: Trending regime + 1d bullish + 4h bullish + CRSI neutral + volume + session
        if trending_regime and trend_1d_bullish and trend_4h_bullish and crsi_neutral_long and volume_ok and in_session:
            long_signal = True
        
        # Path 3: Range regime + 1d bullish + 4h bullish + CRSI very oversold + volume + session
        if ranging_regime and trend_1d_bullish and trend_4h_bullish and crsi_1h[i] < 15 and volume_ok and in_session:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS (3+ confluence required) ===
        short_signal = False
        
        # Path 1: Strong trend regime + 1d/4h bearish + CRSI overbought + volume + session
        if trending_regime and trend_1d_bearish and trend_4h_bearish and crsi_overbought and volume_ok and in_session:
            short_signal = True
        
        # Path 2: Trending regime + 1d bearish + 4h bearish + CRSI neutral + volume + session
        if trending_regime and trend_1d_bearish and trend_4h_bearish and crsi_neutral_short and volume_ok and in_session:
            short_signal = True
        
        # Path 3: Range regime + 1d bearish + 4h bearish + CRSI very overbought + volume + session
        if ranging_regime and trend_1d_bearish and trend_4h_bearish and crsi_1h[i] > 85 and volume_ok and in_session:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals (rare), go with 1d HMA trend
        if long_signal and short_signal:
            if trend_1d_bullish:
                desired_signal = BASE_SIZE
            elif trend_1d_bearish:
                desired_signal = -BASE_SIZE
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
        
        # === EXIT CONDITIONS ===
        # Exit long if HTF trend reverses or CRSI goes overbought
        if in_position and position_side > 0:
            if trend_1d_bearish or trend_4h_bearish:
                desired_signal = 0.0
            elif crsi_1h[i] > 70:  # Take profit on CRSI mean reversion
                desired_signal = 0.0
        
        # Exit short if HTF trend reverses or CRSI goes oversold
        if in_position and position_side < 0:
            if trend_1d_bullish or trend_4h_bullish:
                desired_signal = 0.0
            elif crsi_1h[i] < 30:  # Take profit on CRSI mean reversion
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