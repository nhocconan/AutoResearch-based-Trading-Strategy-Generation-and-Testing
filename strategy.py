#!/usr/bin/env python3
"""
Experiment #420: 1d Connors RSI Mean Reversion + Weekly HMA Trend + Choppiness Regime + ATR Stop
Hypothesis: Connors RSI (CRSI) is superior to standard RSI for mean reversion with 75% win rate.
Combined with Choppiness Index to detect range vs trend regimes, and Weekly HMA for trend bias.
CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3. Long when CRSI<15, short when CRSI>85.
Choppiness > 61.8 = range (enable mean reversion), Choppiness < 38.2 = trend (reduce position).
This should generate MORE trades than Fisher-based strategies (#408 failed with Sharpe=-0.029).
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
Position size: 0.30 max, discrete levels, stoploss 2.5*ATR for daily timeframe.
Target: Beat Sharpe=0.499 with >=10 trades/symbol on train, >=3 on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_weekly_hma_atr_v1"
timeframe = "1d"
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

def calculate_rsi_streak(close, period=2):
    """Calculate RSI Streak component of CRSI.
    Measures consecutive up/down days. +100 for N up days, -100 for N down days.
    """
    n = len(close)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        up_streak = 0
        down_streak = 0
        
        # Count consecutive up days
        for j in range(i, max(0, i - 20), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
            else:
                break
        
        # Count consecutive down days
        for j in range(i, max(0, i - 20), -1):
            if j == 0:
                break
            if close[j] < close[j-1]:
                down_streak += 1
            else:
                break
        
        # Streak RSI formula: 100 * up_streak / (up_streak + down_streak) if up_streak > 0
        # Or -100 * down_streak / (up_streak + down_streak) if down_streak > 0
        total = up_streak + down_streak
        if total > 0:
            if up_streak >= down_streak:
                streak_rsi[i] = 100.0 * up_streak / total
            else:
                streak_rsi[i] = -100.0 * down_streak / total + 100
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank component of CRSI.
    Percentage of closes in last N periods that are lower than current close.
    """
    n = len(close)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])
        pr[i] = 100.0 * count_lower / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + streak_rsi + pr) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/consolidation, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        if highest == lowest or atr_sum == 0:
            chop[i] = 50.0
            continue
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    sma50 = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    SIZE_QUARTER = 0.08
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):  # Start after 150 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma50[i]) or np.isnan(sma200[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (long-term direction)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Choppiness regime detection
        is_range = chop[i] > 55.0  # Range/consolidation (enable mean reversion)
        is_trend = chop[i] < 45.0  # Trending (reduce mean reversion size)
        
        # CRSI extreme levels (mean reversion signals)
        crsi_oversold = crsi[i] < 20.0  # Very oversold
        crsi_overbought = crsi[i] > 80.0  # Very overbought
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # CRSI turning (momentum shift)
        crsi_turning_up = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_turning_down = crsi[i] < crsi[i-1] if i > 0 else False
        
        # Price position relative to SMAs
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: CRSI extreme oversold + range regime + weekly bullish (primary MR)
        if crsi_extreme_oversold and is_range and weekly_bullish:
            new_signal = SIZE_ENTRY
        # Path 2: CRSI oversold + turning up + above SMA200 (trend-aligned MR)
        elif crsi_oversold and crsi_turning_up and above_sma200:
            new_signal = SIZE_ENTRY
        # Path 3: CRSI < 15 + weekly bullish (simplified, more trades)
        elif crsi[i] < 15.0 and weekly_bullish and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 4: CRSI turning up from oversold + range (momentum shift)
        elif crsi[i-1] < 25.0 and crsi_turning_up and is_range and rsi_ok_long(crsi[i]):
            new_signal = SIZE_ENTRY
        # Path 5: Simple - CRSI < 20 + above SMA50 (less restrictive)
        elif crsi_oversold and above_sma50:
            new_signal = SIZE_ENTRY * 0.7  # Smaller size without weekly confirmation
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: CRSI extreme overbought + range regime + weekly bearish (primary MR)
        if crsi_extreme_overbought and is_range and weekly_bearish:
            new_signal = -SIZE_ENTRY
        # Path 2: CRSI overbought + turning down + below SMA200 (trend-aligned MR)
        elif crsi_overbought and crsi_turning_down and below_sma200:
            new_signal = -SIZE_ENTRY
        # Path 3: CRSI > 85 + weekly bearish (simplified, more trades)
        elif crsi[i] > 85.0 and weekly_bearish and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 4: CRSI turning down from overbought + range (momentum shift)
        elif crsi[i-1] > 75.0 and crsi_turning_down and is_range and rsi_ok_short(crsi[i]):
            new_signal = -SIZE_ENTRY
        # Path 5: Simple - CRSI > 80 + below SMA50 (less restrictive)
        elif crsi_overbought and below_sma50:
            new_signal = -SIZE_ENTRY * 0.7  # Smaller size without weekly confirmation
        
        # Reduce position size in trending markets (mean reversion less effective)
        if is_trend and new_signal != 0.0:
            new_signal = new_signal * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for daily timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for daily timeframe)
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

def rsi_ok_long(crsi_val):
    """Helper to check if CRSI is in valid long range."""
    return crsi_val < 50.0

def rsi_ok_short(crsi_val):
    """Helper to check if CRSI is in valid short range."""
    return crsi_val > 50.0