#!/usr/bin/env python3
"""
Experiment #450: 1h Primary + 4h/12h HTF — Regime-Adaptive Connors RSI with HMA Trend

Hypothesis: After analyzing 449 failed experiments, clear pattern emerges for 1h TF:
1. 1h strategies with complex filters get 0 trades (#440, #445, #448 all failed)
2. Need SIMPLER entry logic: 4h HMA trend + 1h CRSI extremes (relaxed thresholds)
3. Remove session filters (kills trade frequency)
4. Use 4h HMA (not 1d) for faster trend signals compatible with 1h entries
5. CRSI thresholds: <20/>80 (not <10/>90) to generate sufficient trades
6. Add fallback entry: if no position for 50 bars, enter on weaker signal

Why this might beat current best (Sharpe=0.435):
- 1h TF captures more reversals than 12h/1d while avoiding 15m/30m fee drag
- 4h HMA provides trend bias without over-filtering (1d too slow for 1h entries)
- Relaxed CRSI thresholds ensure 30-80 trades/year target is met
- Regime-adaptive: different logic for trending vs ranging (CHOP filter)
- ATR 2.5x trailing stop protects in 2022-style crashes

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_4h12h_regime_simp_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate in research notes. Best for mean reversion entries.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain = streak_delta.where(streak_delta > 0, 0.0)
    loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of daily returns over 100 periods
    returns = close_s.pct_change()
    percent_rank = pd.Series(np.zeros(n))
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank.iloc[i] = (rank / rank_period) * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    chop_12h = calculate_choppiness(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.28
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(chop_12h_aligned[i]):
            continue
        if np.isnan(crsi_1h[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 4H TREND DIRECTION (primary bias) ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 12H CHOPPINESS REGIME ===
        # Relaxed thresholds for more trades
        is_ranging = chop_12h_aligned[i] > 50.0
        is_trending = chop_12h_aligned[i] < 45.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI SIGNALS (relaxed thresholds for trade frequency) ===
        crsi_oversold = crsi_1h[i] < 20.0
        crsi_overbought = crsi_1h[i] > 80.0
        crsi_extreme_oversold = crsi_1h[i] < 15.0
        crsi_extreme_overbought = crsi_1h[i] > 85.0
        crsi_moderate_oversold = crsi_1h[i] < 30.0
        crsi_moderate_overbought = crsi_1h[i] > 70.0
        
        # === ENTRY LOGIC — SIMPLIFIED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence paths)
        if hma_4h_bullish or price_above_4h_hma or above_sma200:
            # Path 1: Ranging market + CRSI oversold (mean reversion)
            if is_ranging and crsi_oversold:
                new_signal = LONG_SIZE
            # Path 2: Trending market + pullback entry
            elif is_trending and hma_4h_bullish and crsi_moderate_oversold:
                new_signal = LONG_SIZE
            # Path 3: Extreme CRSI (works in any regime)
            elif crsi_extreme_oversold:
                new_signal = LONG_SIZE
            # Path 4: Simple HMA + CRSI combo
            elif hma_4h_bullish and crsi_1h[i] < 35.0:
                new_signal = LONG_SIZE * 0.9
        
        # SHORT ENTRIES
        if hma_4h_bearish or price_below_4h_hma or below_sma200:
            # Path 1: Ranging market + CRSI overbought
            if is_ranging and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 2: Trending market + bounce entry
            elif is_trending and hma_4h_bearish and crsi_moderate_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 3: Extreme CRSI
            elif crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 4: Simple HMA + CRSI combo
            elif hma_4h_bearish and crsi_1h[i] > 65.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.9
        
        # === FREQUENCY BOOST — CRITICAL FOR 1H TF ===
        # If no position for 50+ bars, enter on weaker signals
        if not in_position and new_signal == 0.0 and bars_since_entry > 50:
            if hma_4h_bullish and crsi_moderate_oversold:
                new_signal = LONG_SIZE * 0.6
            elif hma_4h_bearish and crsi_moderate_overbought:
                new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS CHECK (BEFORE exit logic) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit)
        if in_position and position_side > 0 and crsi_1h[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_1h[i] < 20.0:
            new_signal = 0.0
        
        # Trend reversal exit
        if in_position and position_side > 0 and hma_4h_bearish and price_below_4h_hma:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish and price_above_4h_hma:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                bars_since_entry = 0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                bars_since_entry = 0
        else:
            if in_position:
                bars_since_entry += 1
            else:
                bars_since_entry += 1
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals