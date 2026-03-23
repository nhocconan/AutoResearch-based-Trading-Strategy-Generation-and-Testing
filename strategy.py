#!/usr/bin/env python3
"""
Experiment #1020: 1h Primary + 4h/12h HTF — CRSI + Donchian + Choppiness Regime

Hypothesis: After 739 failed strategies, the pattern is clear:
- 1h strategies fail when entry filters are too strict (0 trades)
- 1h strategies fail when entry filters are too loose (>200 trades/year = fee drag)
- The winning formula: HTF trend (4h/12h HMA) + CRSI pullback + relaxed Donchian

This strategy uses:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 20 (oversold pullback in uptrend)
   - Short: CRSI > 80 (overbought pullback in downtrend)
   - RELAXED from 10/90 to generate MORE trades

2. 4h HMA21 + 12h HMA21: Dual HTF trend filter
   - Long only when price > 4h HMA (medium trend up)
   - Short only when price < 12h HMA (long trend down)
   - Asymmetric for bear market bias

3. DONCHIAN(20) BREAKOUT: Momentum confirmation
   - Long: price breaks above Donchian upper (but CRSI still low = pullback entry)
   - Short: price breaks below Donchian lower (but CRSI still high = pullback entry)

4. CHOPPINESS INDEX: Regime filter (relaxed thresholds)
   - CHOP > 55 = range (favor mean reversion entries)
   - CHOP < 45 = trend (favor breakout entries)
   - Between = allow both (more trades!)

5. ATR Trailing Stop: 2.5x ATR for risk management

6. SESSION FILTER: 6-22 UTC (wider than before for more trades)

7. VOLUME FILTER: Volume > 0.6x 20-bar avg (light filter)

Key differences from failed 1h strategies:
- RELAXED CRSI thresholds (20/80 not 10/90) for MORE trades
- RELAXED Choppiness thresholds (55/45 not 61.8/38.2)
- Session filter is WIDER (6-22 UTC, not 8-20)
- Volume filter is light (0.6x not 1.2x)
- BASE_SIZE = 0.25 (smaller for 1h to reduce drawdown)

Target: 40-80 trades/year, Sharpe > 0.612, all symbols positive
Timeframe: 1h (as specified in experiment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_donchian_4h12h_hma_chop_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on close
    RSI(streak, 2): RSI on up/down streak duration
    PercentRank(100): Percentile rank of today's return over last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Calculate RSI(3) on close
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100 - (100 / (1 + rs))
    rsi_close = rsi_close.values
    
    # Calculate streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_series = pd.Series(streak)
    streak_delta = streak_series.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = (-streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # Calculate PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and returns[-1] != 0:
            count_below = np.sum(returns[:-1] < returns[-1])
            percent_rank[i] = count_below / (len(returns) - 1) * 100
        else:
            percent_rank[i] = 50
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channels: Upper = highest high, Lower = lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    We use relaxed: > 55 = range, < 45 = trend
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range using EMA smoothing."""
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
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA21 for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA21 for long-term trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(chop_1h[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (6-22 UTC for more trades) ===
        open_time = prices["open_time"].iloc[i]
        if isinstance(open_time, (int, np.integer)):
            hour_utc = (open_time // 3600000 + 8) % 24
        else:
            hour_utc = pd.to_datetime(open_time).hour
        
        in_session = 6 <= hour_utc <= 22
        
        # === VOLUME FILTER (light: 0.6x avg) ===
        volume_ok = volume[i] > 0.6 * vol_avg[i]
        
        # === HTF TREND (4h + 12h HMA21) ===
        # Asymmetric: easier to long (4h), harder to short (12h) for bear bias
        medium_bull = close[i] > hma_4h_aligned[i]
        long_bear = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index - relaxed) ===
        regime_range = chop_1h[i] > 55  # Ranging market
        regime_trend = chop_1h[i] < 45  # Trending market
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === CONNORS RSI SIGNALS (relaxed: 20/80 not 10/90) ===
        crsi_oversold = crsi_1h[i] < 20
        crsi_overbought = crsi_1h[i] > 80
        crsi_mild_oversold = crsi_1h[i] < 30
        crsi_mild_overbought = crsi_1h[i] > 70
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if medium_bull and position_side <= 0:
            # Primary: CRSI oversold in bullish medium trend
            if crsi_oversold and in_session and volume_ok:
                desired_signal = BASE_SIZE
            # Secondary: Range market + mild oversold
            elif regime_range and crsi_mild_oversold and in_session:
                desired_signal = REDUCED_SIZE
            # Tertiary: Trend + Donchian breakout + CRSI not extreme
            elif regime_trend and donchian_breakout_up and crsi_1h[i] < 50 and in_session:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if long_bear and position_side >= 0:
            # Primary: CRSI overbought in bearish long trend
            if crsi_overbought and in_session and volume_ok:
                desired_signal = -BASE_SIZE
            # Secondary: Range market + mild overbought
            elif regime_range and crsi_mild_overbought and in_session:
                desired_signal = -REDUCED_SIZE
            # Tertiary: Trend + Donchian breakdown + CRSI not extreme
            elif regime_trend and donchian_breakout_down and crsi_1h[i] > 50 and in_session:
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
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if medium trend reverses + CRSI overbought
            if not medium_bull and crsi_1h[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if long trend reverses + CRSI oversold
            if not long_bear and crsi_1h[i] < 40:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if medium bullish and CRSI not extreme overbought
                if medium_bull and crsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if long bearish and CRSI not extreme oversold
                if long_bear and crsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
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