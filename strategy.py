#!/usr/bin/env python3
"""
Experiment #1148: 30m Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion

Hypothesis: Lower TF (30m) strategies fail due to TOO MANY TRADES → fee drag.
Solution: Use HTF (4h/1d) for SIGNAL DIRECTION, 30m only for ENTRY TIMING.
This combines:
1. 1d HMA(21) for macro trend bias (long only above, short only below)
2. 4h Choppiness Index(14) for regime detection (>55 = range, <45 = trend)
3. 30m Connors RSI for entry timing (CRSI<15 long, CRSI>85 short)
4. Session filter: only 8-20 UTC (high liquidity, avoid Asia overnight)
5. Volume filter: volume > 0.8x 20-bar average
6. ATR(14) 2.5x trailing stoploss

Why this should beat Sharpe=0.612:
- Session filter cuts ~50% of potential trades (avoid low-liquidity whipsaws)
- Volume filter ensures real moves, not fakeouts
- Connors RSI (not regular RSI) has 75% win rate on extremes per research
- HTF regime filter prevents mean-reversion in strong trends (and vice versa)
- Position size 0.25 (smaller for 30m to handle more frequent signals)

Timeframe: 30m (primary)
HTF: 4h (Choppiness), 1d (HMA trend) — loaded ONCE before loop
Position Size: 0.25 base (discrete: 0.0, ±0.25)
Stoploss: 2.5x ATR trailing
Target: 30-80 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_hma_4h1d_session_atr_v2"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
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
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate Choppiness
    range_hl = highest - lowest
    mask = range_hl > 1e-10
    
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(period)
    chop[~mask] = 50.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite momentum indicator for mean reversion.
    Formula: (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(close, 3)
    def rsi(series, period):
        diff = np.diff(series)
        gain = np.where(diff > 0, diff, 0.0)
        loss = np.where(diff < 0, -diff, 0.0)
        gain = np.concatenate([[0.0], gain])
        loss = np.concatenate([[0.0], loss])
        avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
        mask = avg_loss > 1e-10
        rs = np.zeros(n)
        rs[mask] = avg_gain[mask] / avg_loss[mask]
        result = np.full(n, 50.0)
        result[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
        return result
    
    rsi_close = rsi(close, rsi_period)
    
    # RSI(Streak, 2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    rsi_streak = rsi(streak, streak_period)
    
    # PercentRank(100) — where does current close rank in last 100 bars?
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine
    mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_close[mask] + rsi_streak[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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

def calculate_volume_avg(volume, period=20):
    """Rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h Choppiness for regime detection
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Extract hour for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 30m timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            continue
        if atr[i] <= 1e-10 or np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-liquidity hours
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        # Volume must be > 0.8x average
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        # CHOP > 55 = range (use mean reversion)
        # CHOP < 45 = trend (use trend following)
        is_range = chop_4h_aligned[i] > 55.0
        is_trend = chop_4h_aligned[i] < 45.0
        
        # === ENTRY SIGNAL (30m Connors RSI) ===
        # CRSI < 15 = oversold (long), CRSI > 85 = overbought (short)
        crsi_oversold = crsi_30m[i] < 15.0
        crsi_overbought = crsi_30m[i] > 85.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + (range regime + CRSI oversold OR trend regime + pullback)
        # Session + volume filters required
        if in_session and volume_ok:
            if macro_bull and is_range and crsi_oversold:
                # Range regime: mean reversion long
                desired_signal = BASE_SIZE
            elif macro_bull and is_trend and crsi_oversold:
                # Trend regime: buy pullback in uptrend
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + (range regime + CRSI overbought OR trend regime + rally)
        if in_session and volume_ok:
            if macro_bear and is_range and crsi_overbought:
                # Range regime: mean reversion short
                desired_signal = -BASE_SIZE
            elif macro_bear and is_trend and crsi_overbought:
                # Trend regime: short rally in downtrend
                desired_signal = -BASE_SIZE
        
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
                # Hold long if macro still bull
                if macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses
        if in_position and position_side > 0:
            if macro_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
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