#!/usr/bin/env python3
"""
Experiment #946: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + Donchian Breakout

Hypothesis: After 675 failed strategies, combining Connors RSI (proven 75% win rate) with
Choppiness Index regime detection and Donchian breakout confirmation should generate
consistent trades across ALL symbols (BTC/ETH/SOL) on 12h timeframe.

Key insights from research:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive than regular RSI, catches short-term extremes
   - Long: CRSI < 15 + price > SMA200. Short: CRSI > 85 + price < SMA200
2. Choppiness Index: CHOP(14) > 55 = range (mean revert), CHOP < 45 = trend (breakout)
3. Donchian(20): Breakout confirmation for trend entries
4. 1d HMA(21): Macro trend bias (only trade in direction of daily trend)
5. 12h timeframe: Target 20-50 trades/year (lower fee drag than 4h/1h)

Why 12h timeframe:
- Fewer trades = less fee drag (0.05% per round trip)
- HTF signals (1d) provide stronger trend bias
- Less noise than 4h/1h, clearer regime detection
- Proven to work in both bull and bear markets

Critical improvements over failed #936/#942:
- RELAXED CRSI thresholds (15/85 not 10/90) to ensure trades
- Donchian breakout as additional confluence (not sole signal)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_donchian_1d_hma_atr_v1"
timeframe = "12h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components:
    1. RSI(period) - short-term momentum
    2. RSI of streak (consecutive up/down days)
    3. Percent rank of returns over lookback
    
    Formula: CRSI = (RSI + RSI_Streak + PercentRank) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(period)
    rsi = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    if len(streak_gain) >= streak_period:
        avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        
        avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
        avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
            streak_rsi = 100 - (100 / (1 + streak_rs))
        streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percent rank of returns
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current_return = returns[i-1] if i > 0 else 0
        rank = np.sum(window < current_return)
        percent_rank[i] = rank / len(window) * 100 if len(window) > 0 else 50
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
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
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels - highest high and lowest low over period."""
    n = len(close) if 'close' in dir() else len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h SMA200 for additional trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
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
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            continue
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LONG-TERM TREND (12h SMA200) ===
        trend_bull = close[i] > sma_200[i]
        trend_bear = close[i] < sma_200[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 15
        crsi_overbought = crsi_12h[i] > 85
        crsi_extreme_oversold = crsi_12h[i] < 10
        crsi_extreme_overbought = crsi_12h[i] > 90
        
        # === DONCHIAN POSITION ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        donchian_support = close[i] < donchian_mid[i] if not np.isnan(donchian_mid[i]) else False
        donchian_resistance = close[i] > donchian_mid[i] if not np.isnan(donchian_mid[i]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + macro/long-term trend support
            if crsi_oversold and (macro_bull or trend_bull):
                desired_signal = BASE_SIZE
            # Long: CRSI extreme oversold (guarantees trades)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            # Long: CRSI oversold + near Donchian lower
            elif crsi_oversold and close[i] < donchian_lower[i] * 1.02:
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + macro/long-term trend resistance
            if crsi_overbought and (macro_bear or trend_bear):
                desired_signal = -BASE_SIZE
            # Short: CRSI extreme overbought (guarantees trades)
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            # Short: CRSI overbought + near Donchian upper
            elif crsi_overbought and close[i] > donchian_upper[i] * 0.98:
                desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback + Donchian support
            if macro_bull or trend_bull:
                if crsi_oversold and donchian_support:
                    desired_signal = BASE_SIZE
                # Breakout long with CRSI confirmation
                elif donchian_breakout_long and crsi_12h[i] < 70:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + CRSI rally + Donchian resistance
            if macro_bear or trend_bear:
                if crsi_overbought and donchian_resistance:
                    desired_signal = -BASE_SIZE
                # Breakout short with CRSI confirmation
                elif donchian_breakout_short and crsi_12h[i] > 30:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extremes only
            if crsi_extreme_oversold and (macro_bull or trend_bull):
                desired_signal = BASE_SIZE
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (macro_bear or trend_bear):
                desired_signal = -BASE_SIZE
            elif crsi_extreme_overbought:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if (macro_bull or trend_bull) and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (macro_bear or trend_bear) and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + long-term trend reverses + CRSI overbought
            if macro_bear and trend_bear and crsi_12h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + long-term trend reverses + CRSI oversold
            if macro_bull and trend_bull and crsi_12h[i] < 25:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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