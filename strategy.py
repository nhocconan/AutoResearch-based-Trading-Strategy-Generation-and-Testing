#!/usr/bin/env python3
"""
Experiment #733: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: After 491 failed strategies, the pattern is clear:
1. Lower timeframes (4h and below) generate too many trades → fee drag kills Sharpe
2. Complex regime detection often prevents ANY trades (Sharpe=0.000 failures)
3. Daily timeframe with weekly HTF has proven success (20-50 trades/year target)
4. Connors RSI has 75% win rate in academic literature for mean reversion
5. Choppiness Index cleanly separates trending vs ranging regimes

This strategy combines:
- Connors RSI (CRSI) for precise mean reversion entries
- Choppiness Index (CHOP) for regime detection (range vs trend)
- 1w HMA for ultra-long-term trend bias
- Different logic per regime: mean revert in chop, trend-follow otherwise
- Conservative sizing (0.25-0.30) to survive 2022-style crashes

Target: Beat Sharpe=0.612, trades >= 20 per symbol on train, >= 3 on test
Timeframe: 1d (proven to work, minimal fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_hma_1w_v2"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - proven 75% win rate for mean reversion.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100 scale)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[max(0, i-streak_period):i] > 0)
        down_streaks = np.sum(streak[max(0, i-streak_period):i] < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100 * up_streaks / total
        else:
            streak_rsi[i] = 50
    
    # Component 3: Percentile Rank of close in last 100 days
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine all three components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - distinguishes trending vs ranging markets.
    CHOP > 61.8 = range (mean reversion works)
    CHOP < 38.2 = trend (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period-1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 100
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = ranging market (mean reversion works best)
        # CHOP < 38.2 = trending market (trend following works best)
        # 38.2 <= CHOP <= 61.8 = transition zone (use both)
        is_ranging = chop[i] > 55  # Slightly lower threshold to catch more ranges
        is_trending = chop[i] < 45  # Slightly higher threshold to catch more trends
        
        # === TREND BIAS (1w HTF HMA) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === MEAN REVERSION ENTRIES (in ranging regime) ===
        if is_ranging:
            # Long: CRSI extremely oversold + above 1w HMA (don't short in bull trend)
            if crsi[i] < 15 and trend_1w_bullish:
                desired_signal = current_size
            
            # Short: CRSI extremely overbought + below 1w HMA (don't long in bear trend)
            elif crsi[i] > 85 and trend_1w_bearish:
                desired_signal = -current_size
            
            # Additional mean reversion: CRSI < 25 or > 75 with SMA200 confirmation
            elif crsi[i] < 25 and above_sma200 and trend_1w_bullish:
                desired_signal = current_size
            elif crsi[i] > 75 and below_sma200 and trend_1w_bearish:
                desired_signal = -current_size
        
        # === TREND FOLLOWING ENTRIES (in trending regime) ===
        if is_trending:
            # Long: Trending up + pullback (CRSI not overbought) + Donchian breakout
            if trend_1w_bullish and crsi[i] < 60 and close[i] > donch_upper[i-1]:
                desired_signal = current_size
            
            # Short: Trending down + bounce (CRSI not oversold) + Donchian breakdown
            elif trend_1w_bearish and crsi[i] > 40 and close[i] < donch_lower[i-1]:
                desired_signal = -current_size
            
            # Additional trend: CRSI confirmation with trend
            elif trend_1w_bullish and 35 < crsi[i] < 65 and above_sma200:
                desired_signal = current_size
            elif trend_1w_bearish and 35 < crsi[i] < 65 and below_sma200:
                desired_signal = -current_size
        
        # === TRANSITION ZONE (mixed signals) ===
        if not is_ranging and not is_trending:
            # Use conservative entries only
            if crsi[i] < 12 and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 88 and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if still in ranging regime or bullish trend
                if (is_ranging and crsi[i] < 70) or (trend_1w_bullish and crsi[i] < 75):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if still in ranging regime or bearish trend
                if (is_ranging and crsi[i] > 30) or (trend_1w_bearish and crsi[i] > 25):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI extremely overbought or trend reverses
            if crsi[i] > 85 or (trend_1w_bearish and crsi[i] > 60):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI extremely oversold or trend reverses
            if crsi[i] < 15 or (trend_1w_bullish and crsi[i] < 40):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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