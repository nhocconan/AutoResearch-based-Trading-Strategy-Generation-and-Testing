#!/usr/bin/env python3
"""
Experiment #1064: 4h Primary + 12h/1d HTF — Dual Regime with Relaxed Entries

Hypothesis: After analyzing 771+ failed experiments, the key insight is:
1. 4h timeframe balances trade frequency (20-50/year) vs fee drag
2. Connors RSI works BEST in bear/range markets (2022 crash, 2025 bear)
3. Donchian breakout works in trending regimes
4. Choppiness Index regime filter is CRITICAL for switching logic
5. RELAXED thresholds are essential — too strict = 0 trades (see #1055, #1059)
6. 12h HMA21 provides macro bias without being too laggy like 1w

Key changes from failed experiments:
- CRSI thresholds: <20/>80 (not <10/>90) to ensure trades
- CHOP thresholds: >50/<50 (not >61.8/<38.2) for more regime switches
- ADX threshold: >12 (not >25) for trend confirmation
- Position size: 0.25 discrete levels to minimize churn

Timeframe: 4h
HTF: 12h (HMA21 for macro bias)
Position Size: 0.25 base, 0.15 reduced
Stoploss: 2.5x ATR trailing
Target: 30-60 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_crsi_donchian_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_close = 100 - (100 / (1 + rs))
    rsi_close[:rsi_period + 5] = np.nan
    
    # Component 2: RSI on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_series = pd.Series(streak_gain)
    streak_loss_series = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = streak_loss_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = np.divide(avg_streak_gain, avg_streak_loss, out=np.ones_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak[:streak_period + 10] = np.nan
    
    # Component 3: Percent Rank
    percent_rank = np.full(n, np.nan)
    daily_return = np.diff(close) / np.maximum(close[:-1], 1e-10) * 100
    daily_return = np.insert(daily_return, 0, 0)
    
    for i in range(rank_period, n):
        window = daily_return[i - rank_period + 1:i + 1]
        current = daily_return[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index — CHOP > 50 = range, CHOP < 50 = trend"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average"""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels for breakout detection"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, lower
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength"""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    smoothed_plus_dm = plus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_tr = tr_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(100 * smoothed_plus_dm, smoothed_tr, out=np.zeros_like(smoothed_plus_dm), where=smoothed_tr != 0)
    minus_di = np.divide(100 * smoothed_minus_dm, smoothed_tr, out=np.zeros_like(smoothed_minus_dm), where=smoothed_tr != 0)
    
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.divide(100 * di_diff, di_sum, out=np.zeros_like(di_diff), where=di_sum != 0)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA21 for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(adx[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 50.0  # Ranging market (mean reversion)
        is_trend = chop[i] < 50.0  # Trending market (trend following)
        
        # === MACRO TREND (12h HMA21) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION with Connors RSI ===
        if is_range:
            # Long: CRSI oversold + macro bullish bias (RELAXED thresholds)
            if crsi[i] < 20 and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + macro bearish bias
            elif crsi[i] > 80 and macro_bear:
                desired_signal = -BASE_SIZE
            # Weaker signals with reduced size (ensure trades)
            elif crsi[i] < 15:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 85:
                desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: Donchian Breakout ===
        elif is_trend:
            # Long: Price breaks Donchian upper + macro bullish + ADX confirms
            if close[i] > donchian_upper[i - 1] and macro_bull and adx[i] > 12:
                desired_signal = BASE_SIZE
            # Short: Price breaks Donchian lower + macro bearish + ADX confirms
            elif close[i] < donchian_lower[i - 1] and macro_bear and adx[i] > 12:
                desired_signal = -BASE_SIZE
            # Weaker trend signals (ensure trades)
            elif close[i] > donchian_upper[i - 1] and macro_bull:
                desired_signal = REDUCED_SIZE
            elif close[i] < donchian_lower[i - 1] and macro_bear:
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
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish or CRSI not overbought
                if macro_bull or crsi[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish or CRSI not oversold
                if macro_bear or crsi[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish AND CRSI overbought
            if macro_bear and crsi[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish AND CRSI oversold
            if macro_bull and crsi[i] < 30:
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