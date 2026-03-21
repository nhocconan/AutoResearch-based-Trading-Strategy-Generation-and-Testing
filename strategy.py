#!/usr/bin/env python3
"""
Experiment #343: 15m Connors RSI Mean Reversion + 4h HMA Trend Filter + Choppiness Regime
Hypothesis: 15m timeframe is ideal for mean reversion with Connors RSI (75% win rate in literature).
4h HMA provides macro trend bias to avoid counter-trend mean reversion traps.
Choppiness Index filters: CHOP>61.8 = range (enable mean reversion), CHOP<38.2 = trend (reduce size).
This should work in both bull (2021) and bear/range (2022, 2025) markets by adapting to regime.
Timeframe: 15m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 30-60 trades/year, mean reversion with trend filter.
Key insight: CRSI extremes + HTF trend + regime filter = high win rate mean reversion that adapts.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_chop_regime_atr_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: rank of today's return vs last 100 days
    """
    n = len(close)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    avg_streak_g = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_l = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    rs_streak = np.where(avg_streak_l > 0, avg_streak_g / avg_streak_l, 100.0)
    rsi_streak = 100 - 100 / (1 + rs_streak)
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank (100)
    returns = np.diff(close, prepend=close[0]) / np.where(close > 0, np.roll(close, 1), 1)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # CRSI = average of three components
    crsi = (rsi_fast + rsi_streak + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest - lowest
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / np.where(price_range > 0, price_range, 1)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    chop = np.where(np.isnan(chop), 50, chop)  # Default to neutral
    
    return chop

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - mean) / np.where(std > 0, std, 1)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Calculate 15m RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    SIZE_QUARTER = 0.10
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        hma_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_valid and close[i] < hma_4h_aligned[i]
        
        # Choppiness regime
        is_ranging = chop[i] > 55  # Range market (mean reversion works)
        is_trending = chop[i] < 45  # Trend market (reduce mean reversion size)
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 15  # Strong oversold
        crsi_overbought = crsi[i] > 85  # Strong overbought
        crsi_mild_oversold = crsi[i] < 25  # Mild oversold
        crsi_mild_overbought = crsi[i] > 75  # Mild overbought
        
        # Z-score confirmation
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # RSI confirmation
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES (Mean Reversion) ===
        # Primary: CRSI extreme oversold + 4h bullish trend + ranging market
        if crsi_oversold and trend_bullish and is_ranging:
            new_signal = SIZE_FULL
        # Secondary: CRSI mild oversold + 4h bullish + zscore confirmation
        elif crsi_mild_oversold and trend_bullish and zscore_oversold:
            new_signal = SIZE_FULL
        # Tertiary: CRSI oversold + RSI oversold (strong mean reversion signal)
        elif crsi_oversold and rsi_oversold:
            new_signal = SIZE_QUARTER  # Smaller size without trend filter
        
        # === SHORT ENTRIES (Mean Reversion) ===
        # Primary: CRSI extreme overbought + 4h bearish trend + ranging market
        if crsi_overbought and trend_bearish and is_ranging:
            new_signal = -SIZE_FULL
        # Secondary: CRSI mild overbought + 4h bearish + zscore confirmation
        elif crsi_mild_overbought and trend_bearish and zscore_overbought:
            new_signal = -SIZE_FULL
        # Tertiary: CRSI overbought + RSI overbought (strong mean reversion signal)
        elif crsi_overbought and rsi_overbought:
            new_signal = -SIZE_QUARTER  # Smaller size without trend filter
        
        # Reduce size in trending markets (mean reversion less reliable)
        if is_trending and new_signal != 0.0:
            new_signal = new_signal * 0.5  # Half size in trends
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals