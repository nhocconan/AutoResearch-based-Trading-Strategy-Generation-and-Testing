#!/usr/bin/env python3
"""
Experiment #1080: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness + HTF HMA Trend

Hypothesis: For 1h timeframe, the winning pattern combines:
1. CONNORS RSI (CRSI) — proven mean reversion signal with 75% win rate
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 15 | Short: CRSI > 85
2. CHOPPINESS INDEX (CHOP) — regime filter to avoid trend-chop mismatch
   CHOP > 55 = range (favor mean reversion) | CHOP < 45 = trend (reduce mean reversion)
3. 4h HMA21 + 12h HMA21 — dual HTF trend filter for direction bias
   Only long if price > 4h HMA AND 4h HMA > 12h HMA
   Only short if price < 4h HMA AND 4h HMA < 12h HMA
4. Session filter (8-20 UTC) — only trade during high liquidity hours
5. Volume filter — volume > 0.8x 20-period average
6. ATR stoploss (2.5x) — mandatory risk management

Why this should work on 1h:
- CRSI is proven for crypto mean reversion (different from failed simple RSI)
- Dual HTF trend filter prevents counter-trend trades (major failure mode)
- Session + volume filters reduce trade count to target 30-80/year
- 1h entries within 4h/12h trend = HTF frequency with LTF precision
- Position size 0.25 (conservative for lower TF)

Timeframe: 1h (primary)
HTF: 4h + 12h — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_dual_htf_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — combines 3 components for mean reversion signal.
    
    Formula:
    CRSI = (RSI(close, 3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days streak
    PercentRank: percentile rank of 1-day price change over rank_period
    
    Signals:
    - CRSI < 10-15 = oversold (long opportunity)
    - CRSI > 85-90 = overbought (short opportunity)
    
    Research shows 75% win rate on crypto mean reversion.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.full(n, 50.0)
    valid_mask = avg_loss > 1e-10
    rsi[valid_mask] = 100.0 - (100.0 / (1.0 + avg_gain[valid_mask] / avg_loss[valid_mask]))
    rsi[~valid_mask] = 100.0
    
    # Component 2: RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, 50.0)
    valid_mask = avg_streak_loss > 1e-10
    rsi_streak[valid_mask] = 100.0 - (100.0 / (1.0 + avg_streak_gain[valid_mask] / avg_streak_loss[valid_mask]))
    rsi_streak[~valid_mask] = 100.0
    
    # Component 3: PercentRank of 1-day returns
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        current_return = returns[i]
        rank = np.sum(window < current_return)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine components
    valid_mask = (~np.isnan(rsi)) & (~np.isnan(rsi_streak)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range market (mean reversion favored)
    - CHOP < 38.2 = trending market (breakout/trend follow favored)
    - 38.2 - 61.8 = transition zone
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA21 for primary trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA21 for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter (8-20 UTC)
    hours = extract_hour(open_time)
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === FILTERS ===
        session_ok = in_session[i]
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === HTF TREND BIAS (Dual HMA) ===
        # Bullish: price > 4h HMA AND 4h HMA > 12h HMA
        trend_bull = (close[i] > hma_4h_aligned[i]) and (hma_4h_aligned[i] > hma_12h_aligned[i])
        # Bearish: price < 4h HMA AND 4h HMA < 12h HMA
        trend_bear = (close[i] < hma_4h_aligned[i]) and (hma_4h_aligned[i] < hma_12h_aligned[i])
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market (favor mean reversion)
        is_trending = chop[i] < 45.0  # Trend market (reduce mean reversion)
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        desired_signal = 0.0
        
        # === MEAN REVERSION ENTRIES (Primary Strategy) ===
        # Long: CRSI oversold + bullish HTF trend + choppy regime + session + volume
        if crsi_oversold and trend_bull and session_ok and volume_ok:
            if is_choppy:
                desired_signal = BASE_SIZE
            elif not is_trending:
                desired_signal = BASE_SIZE * 0.7  # Reduce in trending market
        
        # Short: CRSI overbought + bearish HTF trend + choppy regime + session + volume
        elif crsi_overbought and trend_bear and session_ok and volume_ok:
            if is_choppy:
                desired_signal = -BASE_SIZE
            elif not is_trending:
                desired_signal = -BASE_SIZE * 0.7  # Reduce in trending market
        
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
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI still low or trend still bullish
                if crsi[i] < 50.0 and trend_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI still high or trend still bearish
                if crsi[i] > 50.0 and trend_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought OR trend reverses
            if crsi_overbought or (not trend_bull and crsi[i] > 60.0):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold OR trend reverses
            if crsi_oversold or (not trend_bear and crsi[i] < 40.0):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= BASE_SIZE * 0.5:
                desired_signal = BASE_SIZE * 0.5
            else:
                desired_signal = 0.0
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -BASE_SIZE * 0.5:
                desired_signal = -BASE_SIZE * 0.5
            else:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals