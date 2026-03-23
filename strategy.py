#!/usr/bin/env python3
"""
Experiment #963: 1d Primary + 1w HTF — Connors RSI Mean Reversion + Weekly Trend Filter

Hypothesis: After 691 failed strategies, returning to proven mean reversion with strong
HTF trend filter should work. Connors RSI has 75% win rate in research literature.

Key insights:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Extreme values (<10 long, >90 short) catch reversals
   - Works in both bull and bear markets
2. 1w HMA(21) for macro trend bias — only trade with weekly trend
3. 1d timeframe = 20-50 trades/year target (low fee drag)
4. ATR(14) stoploss at 2.5x for risk management
5. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Why this should beat Sharpe=0.612:
- Simpler logic than failed multi-regime strategies (954, 956, 960, 961, 962)
- Connors RSI proven edge (ETH Sharpe +0.923 in research)
- 1w HTF stronger than 1d/12h used in failed strategies
- Daily bars = clearer signals, less noise than 4h/1h

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_rsi_1w_hma_trend_atr_v1"
timeframe = "1d"
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
    """RSI Streak: consecutive up/down days."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return streak_rsi
    
    # Calculate streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    for i in range(period, n):
        up_streak = max(0, streak[i])
        down_streak = max(0, -streak[i])
        
        if up_streak + down_streak > 0:
            streak_rsi[i] = 100 * up_streak / (up_streak + down_streak + 1e-10)
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank: where current return ranks vs last N days."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period + 1:
        return pr
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current)
        pr[i] = 100 * rank / period
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period + 1:
        return crsi
    
    rsi_3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    pr = calculate_percent_rank(close, period=pr_period)
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pr[i]) / 3.0
    
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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align 1w HMA for macro trend bias
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
    
    for i in range(250, n):  # Need 200 for SMA + 100 for Connors RSI + buffer
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        
        # === MACRO TREND (1w HTF HMA21) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # === LONG-TERM FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Relaxed from 10 to ensure trades
        crsi_overbought = crsi[i] > 85  # Relaxed from 90 to ensure trades
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Connors RSI extreme oversold + weekly bullish trend
        if crsi_extreme_oversold and weekly_bullish:
            desired_signal = BASE_SIZE
        # Secondary: Connors RSI oversold + weekly bullish + above SMA200
        elif crsi_oversold and weekly_bullish and above_sma200:
            desired_signal = REDUCED_SIZE
        # Tertiary: Connors RSI oversold alone (ensures trade frequency)
        elif crsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY ===
        # Primary: Connors RSI extreme overbought + weekly bearish trend
        if crsi_extreme_overbought and weekly_bearish:
            desired_signal = -BASE_SIZE
        # Secondary: Connors RSI overbought + weekly bearish + below SMA200
        elif crsi_overbought and weekly_bearish and below_sma200:
            desired_signal = -REDUCED_SIZE
        # Tertiary: Connors RSI overbought alone (ensures trade frequency)
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
                # Hold long if weekly trend intact and CRSI not overbought
                if weekly_bullish and crsi[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly trend intact and CRSI not oversold
                if weekly_bearish and crsi[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses + CRSI overbought
            if weekly_bearish and crsi[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses + CRSI oversold
            if weekly_bullish and crsi[i] < 20:
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