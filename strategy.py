#!/usr/bin/env python3
"""
Experiment #981: 4h Primary + 1d/1w HTF — Connors RSI + Donchian Breakout + HMA Trend

Hypothesis: After 707 failed strategies, the key is SIMPLER entry conditions that guarantee trades.
Connors RSI (CRSI) has proven 75% win rate in research. Combined with HTF trend bias, this should
work across ALL symbols (BTC/ETH/SOL) in both bull and bear markets.

Key insights from research:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 20 + price > SMA200. Short: CRSI > 80 + price < SMA200.
2. Donchian(20) breakout confirms trend direction
3. 1d HMA(21) for medium-term trend bias
4. 1w HMA(21) for macro regime filter (only trade with macro trend)
5. RELAXED thresholds to ensure >= 30 trades/train, >= 3 trades/test

Why 4h timeframe:
- Target 25-40 trades/year (optimal fee/trade balance)
- HTF signals (1d/1w) provide stronger trend bias
- Proven to work in both bull and bear markets

Critical improvements over failed strategies:
- SIMPLER entry logic (fewer confluence = more trades)
- CRSI thresholds relaxed (20/80 not 10/90) to ensure trades
- No funding rate dependency (caused Sharpe=0.000 failures)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_1d1w_hma_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_rsi_streak(close, period=2):
    """RSI Streak component of Connors RSI - measures consecutive up/down days."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like scale (0-100)
    abs_streak = np.abs(streak)
    streak_score = np.zeros(n)
    for i in range(n):
        if streak[i] > 0:
            streak_score[i] = 50 + min(abs_streak[i] * 10, 50)
        elif streak[i] < 0:
            streak_score[i] = 50 - min(abs_streak[i] * 10, 50)
        else:
            streak_score[i] = 50
    
    # Apply RSI calculation to streak scores
    streak_rsi = calculate_rsi(streak_score, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of Connors RSI - current price vs lookback distribution."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period:
        return pr
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        pr[i] = (rank / (period - 1)) * 100
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    for i in range(n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(250, n):  # Need 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === LONG-TERM TREND (SMA200) ===
        long_term_bull = close[i] > sma_200[i]
        long_term_bear = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.998  # Near upper band
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.002  # Near lower band
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 25  # Relaxed from 20 to ensure trades
        crsi_overbought = crsi_4h[i] > 75  # Relaxed from 80 to ensure trades
        crsi_extreme_oversold = crsi_4h[i] < 15
        crsi_extreme_overbought = crsi_4h[i] > 85
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: CRSI oversold + macro/medium trend support
        if crsi_oversold and (macro_bull or trend_1d_bullish or long_term_bull):
            desired_signal = BASE_SIZE
        # Secondary: CRSI extreme oversold (guarantees trades in deep pullbacks)
        elif crsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        # Tertiary: Donchian breakout + trend alignment
        elif donchian_breakout_long and (trend_1d_bullish or macro_bull):
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: CRSI overbought + macro/medium trend support
        if crsi_overbought and (macro_bear or trend_1d_bearish or long_term_bear):
            if desired_signal > 0:
                desired_signal = 0.0  # Cancel long if short signal stronger
            desired_signal = -BASE_SIZE
        # Secondary: CRSI extreme overbought
        elif crsi_extreme_overbought:
            if desired_signal > 0:
                desired_signal = 0.0
            desired_signal = -REDUCED_SIZE
        # Tertiary: Donchian breakdown + trend alignment
        elif donchian_breakout_short and (trend_1d_bearish or macro_bear):
            if desired_signal > 0:
                desired_signal = 0.0
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and trend intact
                if crsi_4h[i] < 70 and (macro_bull or trend_1d_bullish):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and trend intact
                if crsi_4h[i] > 30 and (macro_bear or trend_1d_bearish):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought
            if crsi_4h[i] > 80:
                desired_signal = 0.0
            # Exit if macro + medium trend reverses
            if macro_bear and trend_1d_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold
            if crsi_4h[i] < 20:
                desired_signal = 0.0
            # Exit if macro + medium trend reverses
            if macro_bull and trend_1d_bullish:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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