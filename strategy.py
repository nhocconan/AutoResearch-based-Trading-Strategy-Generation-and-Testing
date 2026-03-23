#!/usr/bin/env python3
"""
Experiment #1051: 4h Primary + 1d/1w HTF — Simplified Regime + Connors RSI + Donchian

Hypothesis: After analyzing 761+ failed experiments, the key issue is OVER-FILTERING.
Strategies with too many confluence conditions generate 0 trades. The winning approach:

1. SIMPLER REGIME LOGIC: Just 2 regimes (trend vs range) with wider transition zone
2. CONNORS RSI: Better mean reversion signal than regular RSI (75% win rate in research)
3. DONCHIAN BREAKOUT: Clean trend entry signal (20-period high/low break)
4. RELAXED THRESHOLDS: RSI 25-75 (not 30-70), CHOP 50-60 (not 55-65)
5. 1d HMA21 + 1w HMA21: Dual HTF macro filter (less restrictive than single)

Key changes from #1044:
- Connors RSI instead of regular RSI (more sensitive to reversals)
- Donchian breakout for trend entries (cleaner than HMA crossover)
- Wider regime thresholds (more trades)
- Simpler hold logic (maintain position longer)
- Entry on pullback in trend mode (not just breakout)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
Position Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simplified_regime_crsi_donchian_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Long: CRSI < 10 (extreme oversold)
    Short: CRSI > 90 (extreme overbought)
    Entry zone: CRSI < 20 or > 80 for more trades
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) on close
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
    rsi_close[:rsi_period] = np.nan
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    streak_gain_series = pd.Series(streak_gain)
    streak_loss_series = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = streak_loss_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = np.divide(avg_streak_gain, avg_streak_loss, out=np.zeros_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak[:streak_period+5] = np.nan
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3
    
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

def calculate_hma(series, period):
    """Hull Moving Average - faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout detection."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    if n < period:
        return upper, lower, middle
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA21 for macro trend filters
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Wider transition zone for more trades
        is_range = chop[i] > 50.0  # Ranging market (mean reversion)
        is_trend = chop[i] < 55.0  # Trending market (trend following)
        # Zone 50-55: can trade both modes
        
        # === MACRO TREND FILTERS (1d HMA + 1w HMA) ===
        # Less restrictive: only need ONE HTF aligned for direction
        macro_bull = close[i] > hma_1d_aligned[i] or close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i] or close[i] < hma_1w_aligned[i]
        
        # Strong macro: both HTF agree
        strong_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        strong_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION with Connors RSI ===
        if is_range:
            # Long: CRSI oversold + some macro support
            if crsi[i] < 20 and macro_bull:
                desired_signal = BASE_SIZE
            elif crsi[i] < 15:  # Extreme oversold, enter even without macro
                desired_signal = REDUCED_SIZE
            # Short: CRSI overbought + some macro resistance
            elif crsi[i] > 80 and macro_bear:
                desired_signal = -BASE_SIZE
            elif crsi[i] > 85:  # Extreme overbought, enter even without macro
                desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: DONCHIAN BREAKOUT + PULLBACK ===
        if is_trend:
            # Long: Price breaks Donchian upper + strong bullish macro
            if close[i] >= donchian_upper[i] * 0.998 and strong_bull:
                desired_signal = BASE_SIZE
            # Long pullback: Price near Donchian middle + CRSI recovering + bullish
            elif close[i] > donchian_middle[i] and crsi[i] < 50 and crsi[i] > 30 and macro_bull:
                desired_signal = REDUCED_SIZE
            # Short: Price breaks Donchian lower + strong bearish macro
            elif close[i] <= donchian_lower[i] * 1.002 and strong_bear:
                desired_signal = -BASE_SIZE
            # Short pullback: Price near Donchian middle + CRSI declining + bearish
            elif close[i] < donchian_middle[i] and crsi[i] > 50 and crsi[i] < 70 and macro_bear:
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
        
        # === HOLD LOGIC — Maintain position if thesis intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought OR macro still bullish
                if crsi[i] < 75 or macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold OR macro still bearish
                if crsi[i] > 25 or macro_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI extremely overbought
            if crsi[i] > 85:
                desired_signal = 0.0
            # Exit long if strong bearish macro reversal
            if strong_bear and crsi[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI extremely oversold
            if crsi[i] < 15:
                desired_signal = 0.0
            # Exit short if strong bullish macro reversal
            if strong_bull and crsi[i] < 40:
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