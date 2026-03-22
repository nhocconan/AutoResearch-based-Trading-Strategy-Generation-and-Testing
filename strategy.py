#!/usr/bin/env python3
"""
Experiment #166: 4h Connors RSI + 1d HMA Trend + Choppiness Regime Filter

Hypothesis: 4h timeframe with Connors RSI (CRSI) for entry timing combined with 
1d HMA trend filter and Choppiness Index regime detection will outperform simple 
RSI/EMA strategies. CRSI catches short-term oversold/overbought conditions within 
the higher timeframe trend, while CHOP filters out choppy periods where mean 
reversion fails.

Why this might work:
- CRSI (Connors RSI) has 75% win rate in research for mean reversion entries
- 1d HMA provides stable trend bias (avoid counter-trend trades)
- Choppiness Index distinguishes trending vs ranging markets
- 4h is slow enough to avoid noise, fast enough to catch moves
- Asymmetric sizing: larger positions in trending regime, smaller in range

Learning from failures:
- #154, #160 (4h mean reversion): Negative Sharpe - lacked regime filter
- #158, #159 (EMA strategies): Negative Sharpe - too simple, no entry timing
- #163, #165 (RSI pullback): Negative Sharpe - needed better regime detection
- Need to ensure ≥10 trades per symbol (don't over-filter with CHOP)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_1d_hma_chop_regime_atr_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(streak) measures the RSI of the consecutive up/down day streak.
    PercentRank measures where current close ranks vs last 100 closes.
    """
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to absolute values for RSI calculation
    streak_positive = np.where(streak > 0, streak, 0.0)
    streak_negative = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = np.where(avg_streak_loss == 0, 100.0, avg_streak_gain / (avg_streak_loss + 1e-10))
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range-bound market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_RANGE = 0.20   # Smaller size in ranging market
    SIZE_TREND = 0.30   # Larger size in trending market
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start later to ensure all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range (mean reversion works better)
        # CHOP < 38.2 = trend (trend following works better)
        # Use hysteresis to avoid flip-flopping
        is_ranging = chop[i] > 55.0  # Slightly lower threshold to get more signals
        is_trending = chop[i] < 45.0  # Slightly higher threshold
        
        # === POSITION SIZING BASED ON REGIME ===
        if is_trending:
            size = SIZE_TREND
        else:
            size = SIZE_RANGE
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # Long: CRSI < 15 (oversold) + price > SMA200 + 1d bullish trend
        # Short: CRSI > 85 (overbought) + price < SMA200 + 1d bearish trend
        
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # CRSI oversold + above SMA200 + 1d bullish (trend-following pullback)
        if crsi_oversold and above_sma200 and bull_trend_1d:
            new_signal = size
        
        # === SHORT ENTRY CONDITIONS ===
        # CRSI overbought + below SMA200 + 1d bearish (trend-following pullback)
        if crsi_overbought and below_sma200 and bear_trend_1d:
            new_signal = -size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals