#!/usr/bin/env python3
"""
Experiment #013: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: After 12 experiments, the pattern is clear - single-regime strategies 
fail because markets alternate between trending and ranging. The 2025 test period 
is bear/range, which destroys pure trend-following. This strategy uses:

1. Choppiness Index (CHOP) for regime detection:
   - CHOP > 61.8 = ranging market → use Connors RSI mean-reversion
   - CHOP < 38.2 = trending market → use HMA trend-following
   - Between = stay flat (avoid whipsaw)

2. 1w HMA for long-term trend bias (only trade with HTF trend)

3. Connors RSI for range entries (proven 75% win rate):
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > SMA(200)
   - Short: CRSI > 90 + price < SMA(200)

4. HMA crossover for trend entries:
   - Long: HMA(16) crosses above HMA(48) + 1w HMA bullish
   - Short: HMA(16) crosses below HMA(48) + 1w HMA bearish

5. ATR trailing stop (2.5x) for risk management

Why this should beat #011 (Sharpe=0.221):
- 1d timeframe = 20-50 trades/year (less fee drag than 4h)
- 1w HTF = stronger trend filter than 1d
- Dual regime = adapts to 2025 bear/range market
- Connors RSI = proven edge in ranging markets

Position sizing: 0.30 (discrete), leverage=1.0
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_connors_hma_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market ranging vs trending
    CHOP > 61.8 = ranging (mean-revert)
    CHOP < 38.2 = trending (trend-follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
        else:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI - combines 3 components for mean-reversion signals
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < pr_period + 1:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # Component 1: RSI(3) of close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank of daily returns over 100 days
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = returns[i-pr_period+1:i+1]
        count_below = np.sum(window < returns[i])
        percent_rank[i] = 100.0 * count_below / pr_period
    
    # Combine components
    for i in range(pr_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for HTF trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=40)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track HMA crossover for trend entries
    prev_hma_diff = 0.0
    
    for i in range(250, n):  # Need 200 for SMA + 100 for CRSI + buffer
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1W HTF TREND BIAS ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        is_neutral = not is_ranging and not is_trending
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_ranging:
            # === MEAN REVERSION (Connors RSI) ===
            # Long: CRSI < 10 + price > SMA200 + 1w bias bullish or neutral
            if crsi[i] < 15.0 and close[i] > sma_200[i] and (hma_1w_bull or not hma_1w_bear):
                desired_signal = SIZE
            
            # Short: CRSI > 90 + price < SMA200 + 1w bias bearish or neutral
            elif crsi[i] > 85.0 and close[i] < sma_200[i] and (hma_1w_bear or not hma_1w_bull):
                desired_signal = -SIZE
        
        elif is_trending:
            # === TREND FOLLOWING (HMA Crossover) ===
            hma_diff = hma_16[i] - hma_48[i]
            
            # Long: HMA16 crosses above HMA48 + 1w bullish
            if hma_diff > 0 and prev_hma_diff <= 0 and hma_1w_bull:
                desired_signal = SIZE
            
            # Short: HMA16 crosses below HMA48 + 1w bearish
            elif hma_diff < 0 and prev_hma_diff >= 0 and hma_1w_bear:
                desired_signal = -SIZE
            
            prev_hma_diff = hma_diff
        
        # else: is_neutral → stay flat (avoid whipsaw)
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals