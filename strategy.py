#!/usr/bin/env python3
"""
Experiment #429: 1h Connors RSI Mean Reversion + 4h HMA Trend + Choppiness Regime Filter
Hypothesis: Connors RSI (CRSI) mean reversion with 75% win rate works well in bear/range markets.
Combined with 4h HMA for trend bias and Choppiness Index for regime detection, this should
generate frequent signals (>=10 trades/symbol) while maintaining positive Sharpe in 2025 test.
Key insight: Mean reversion outperforms trend-following in bear markets (2022 crash, 2025 range).
CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3. Entry when CRSI<10 (long) or >90 (short).
Timeframe: 1h (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR, take profit at 2R.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_chop_regime_mean_reversion_atr_v1"
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
    Calculate Connors RSI (CRSI) for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Entry: CRSI < 10 (oversold long), CRSI > 90 (overbought short)
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak_rsi = np.zeros(n)
    up_streak = np.zeros(n)
    down_streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            up_streak[i] = up_streak[i-1] + 1
            down_streak[i] = 0
        elif close[i] < close[i-1]:
            down_streak[i] = down_streak[i-1] + 1
            up_streak[i] = 0
        else:
            up_streak[i] = up_streak[i-1]
            down_streak[i] = down_streak[i-1]
    
    # Calculate streak RSI
    for i in range(streak_period, n):
        streak_values = np.zeros(streak_period)
        for j in range(streak_period):
            if up_streak[i-j] > 0:
                streak_values[j] = up_streak[i-j]
            else:
                streak_values[j] = -down_streak[i-j]
        
        # Convert streak to RSI-like value (0-100)
        avg_streak = np.mean(streak_values)
        if avg_streak > 0:
            streak_rsi[i] = 100 * avg_streak / (avg_streak + 1)
        elif avg_streak < 0:
            streak_rsi[i] = 100 * (-avg_streak) / ((-avg_streak) + 1)
        else:
            streak_rsi[i] = 50
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - position in recent range
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index for regime detection.
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        window_high = high[i-period+1:i+1]
        window_low = low[i-period+1:i+1]
        window_close = close[i-period+1:i+1]
        
        highest_high = np.max(window_high)
        lowest_low = np.min(window_low)
        
        if highest_high == lowest_low:
            choppiness[i] = 100
            continue
        
        # Sum of ATR over period
        tr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(
                high[j] - low[j],
                abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j]
            )
            tr_sum += tr
        
        atrl = tr_sum / period
        choppiness[i] = 100 * np.log10((highest_high - lowest_low) / (atrl * np.sqrt(period))) / np.log10(period)
    
    return choppiness

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma50 = calculate_sma(close, 50)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):  # Start after 150 bars for CRSI calculation
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma50[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (long-term direction)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # Choppiness regime filter
        is_ranging = chop[i] > 55  # Mean reversion regime
        is_trending = chop[i] < 45  # Trend following regime
        
        # CRSI mean reversion signals
        crsi_oversold = crsi[i] < 15  # Long entry
        crsi_overbought = crsi[i] > 85  # Short entry
        
        # CRSI extreme (stronger signal)
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # Price position relative to SMA50
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # RSI confirmation (14-period)
        rsi_14 = calculate_rsi(close[i-14:i+1], 14)[-1] if i >= 14 else 50
        rsi_ok_long = rsi_14 < 70
        rsi_ok_short = rsi_14 > 30
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: CRSI oversold + Ranging market + 4h trend bullish (primary mean reversion)
        if crsi_oversold and is_ranging and trend_bullish:
            new_signal = SIZE_ENTRY
        # Path 2: CRSI extreme oversold + Above SMA50 (strong mean reversion)
        elif crsi_extreme_oversold and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 3: CRSI oversold + 4h trend bullish (trend-aligned mean reversion)
        elif crsi_oversold and trend_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 4: CRSI < 20 + Ranging + Price > SMA50 (conservative)
        elif crsi[i] < 20 and is_ranging and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 5: Simple CRSI extreme (ensure trades in all regimes)
        elif crsi_extreme_oversold and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: CRSI overbought + Ranging market + 4h trend bearish (primary mean reversion)
        if crsi_overbought and is_ranging and trend_bearish:
            new_signal = -SIZE_ENTRY
        # Path 2: CRSI extreme overbought + Below SMA50 (strong mean reversion)
        elif crsi_extreme_overbought and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 3: CRSI overbought + 4h trend bearish (trend-aligned mean reversion)
        elif crsi_overbought and trend_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 4: CRSI > 80 + Ranging + Price < SMA50 (conservative)
        elif crsi[i] > 80 and is_ranging and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple CRSI extreme (ensure trades in all regimes)
        elif crsi_extreme_overbought and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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