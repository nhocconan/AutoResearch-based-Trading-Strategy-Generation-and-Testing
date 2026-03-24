#!/usr/bin/env python3
"""
Experiment #1483: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Weekly Trend Filter

Hypothesis: After analyzing 1105+ failed strategies, clear patterns emerge:
1. Complex regime-switching (Choppiness, dual-regime) consistently fails (Sharpe < 0)
2. 4h timeframe struggles with whipsaws and over-filtering (0 trades common)
3. 1d + 1w combination works best for trend-following with fewer trades
4. Connors RSI (CRSI) has proven 75% win rate for mean reversion in bear/range markets
5. Weekly HMA provides robust macro trend filter without over-complication

This strategy combines:
- 1w HMA(21) for macro trend direction (call get_htf_data ONCE!)
- Connors RSI (CRSI) for precise entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 15 + price > weekly_HMA (oversold in uptrend)
- Short: CRSI > 85 + price < weekly_HMA (overbought in downtrend)
- ATR(14)*2.5 trailing stoploss for risk management
- Discrete signal sizes (0.0, ±0.25, ±0.30) to minimize fee churn

Why CRSI works better than standard RSI:
1. RSI(3) captures very short-term momentum extremes
2. RSI_Streak(2) measures consecutive up/down day strength
3. PercentRank(100) shows current price vs 100-day history
4. Combined = more precise oversold/overbought detection than RSI(14)

Timeframe: 1d (target 20-50 trades/year, minimal fee drag ~1-2.5%)
HTF: 1w (call get_htf_data ONCE before loop!)
Position Size: 0.30 max (discrete levels)
Target: Beat Sharpe=0.618, ALL symbols Sharpe > 0, trades >= 30 train / >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - smoother and more responsive than EMA
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper function for WMA
    def wma(series, window):
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights, mode='valid')
        return np.concatenate([np.full(window - 1, np.nan), result])
    
    close_series = pd.Series(close)
    wma_half = wma(close, period // 2)
    wma_full = wma(close, period)
    
    # Handle alignment
    if len(wma_half) > len(wma_full):
        wma_half = wma_half[:len(wma_full)]
    elif len(wma_full) > len(wma_half):
        wma_full = wma_full[:len(wma_half)]
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, int(np.sqrt(period)))
    
    # Pad to match original length
    if len(hma) < n:
        hma = np.concatenate([np.full(n - len(hma), np.nan), hma])
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak Component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate streak: +1 for up day, -1 for down day, 0 for flat
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(period, n):
        if np.isnan(streak[i]):
            continue
        # Calculate average streak over period
        streak_sum = 0
        count = 0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(streak[j]):
                streak_sum += max(0, streak[j])  # Only count positive streaks for RSI
                count += 1
        if count > 0:
            avg_streak = streak_sum / count
            # Normalize to 0-100 scale (typical streak range -5 to +5)
            streak_rsi[i] = min(100, max(0, 50 + avg_streak * 10))
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component of Connors RSI
    Shows where current price ranks vs last 'period' days
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan)
    for i in range(period, n):
        if np.isnan(close[i]):
            continue
        # Count how many of last 'period' closes are below current close
        window = close[i - period + 1:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) < period:
            continue
        count_below = np.sum(valid_window[:-1] < close[i])  # Exclude current from comparison
        pr[i] = (count_below / (len(valid_window) - 1)) * 100
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(n, np.nan)
    for i in range(pr_period, n):
        if np.isnan(rsi_short[i]) or np.isnan(rsi_streak[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average for additional trend filter"""
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
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after all indicators are ready (CRSI needs 100+ bars)
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - direction bias ===
        # Only trade in direction of weekly trend
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === SMA 200 FILTER - additional trend confirmation ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = deeply oversold, CRSI > 85 = deeply overbought
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # Moderate signals for partial entries
        crsi_moderate_oversold = crsi[i] < 25.0
        crsi_moderate_overbought = crsi[i] > 75.0
        
        # === DESIRED SIGNAL - MEAN REVERSION WITH TREND FILTER ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + SMA200 bull + CRSI oversold
        if weekly_bull and above_sma200:
            if crsi_oversold:
                desired_signal = BASE_SIZE  # Strong oversold in uptrend
            elif crsi_moderate_oversold and crsi[i] < crsi[i-1]:
                desired_signal = BASE_SIZE * 0.7  # Moderate + declining CRSI
        
        # SHORT: Weekly bear + SMA200 bear + CRSI overbought
        elif weekly_bear and below_sma200:
            if crsi_overbought:
                desired_signal = -BASE_SIZE  # Strong overbought in downtrend
            elif crsi_moderate_overbought and crsi[i] > crsi[i-1]:
                desired_signal = -BASE_SIZE * 0.7  # Moderate + rising CRSI
        
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.5
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