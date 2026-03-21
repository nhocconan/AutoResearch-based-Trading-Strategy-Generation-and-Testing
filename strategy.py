#!/usr/bin/env python3
"""
Experiment #447: 1h CRSI Mean Reversion + 4h HMA Bias + Choppiness Regime Filter
Hypothesis: Connors RSI (CRSI) has proven 75% win rate for mean reversion. Combined with
4h HMA for trend bias and Choppiness Index for regime detection, this should work well
in the current bear/range market (2025 test period). In range markets (CHOP>61.8), use
CRSI extremes for mean reversion. In trend markets (CHOP<38.2), use pullback entries.
Multiple entry paths ensure >=10 trades per symbol. 2.5*ATR stoploss for 1h timeframe.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_chop_regime_atr_v1"
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
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak = 0
        if close[i] > close[i-1]:
            for j in range(i, max(0, i-streak_period*5), -1):
                if j == 0 or close[j] <= close[j-1]:
                    break
                streak += 1
        elif close[i] < close[i-1]:
            for j in range(i, max(0, i-streak_period*5), -1):
                if j == 0 or close[j] >= close[j-1]:
                    break
                streak -= 1
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            streak_rsi[i] = min(100, 50 + streak * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak * 10)
    
    # Percent Rank (100) - where current return ranks vs last 100 bars
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns) * 100
            pct_rank[i] = rank
    
    # Combine into CRSI
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period:i+1])
        lowest = np.min(low[i-period:i+1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 50.0
    
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return sma, upper, lower

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
    chop = calculate_choppiness(high, low, close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):  # Start after 150 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_bullish = close[i] > hma_4h_aligned[i]
        hma_bearish = close[i] < hma_4h_aligned[i]
        
        # Choppiness regime detection
        is_ranging = chop[i] > 55.0  # Slightly lower threshold for more trades
        is_trending = chop[i] < 45.0
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 25.0  # Looser threshold for more trades
        crsi_overbought = crsi[i] > 75.0
        
        # Bollinger Bands position
        near_lower_bb = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        near_upper_bb = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        below_bb_sma = close[i] < bb_sma[i]
        above_bb_sma = close[i] > bb_sma[i]
        
        # RSI filter
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: CRSI oversold + 4h HMA bullish + Ranging market (mean reversion)
        if crsi_oversold and hma_bullish and is_ranging:
            new_signal = SIZE_ENTRY
        # Path 2: CRSI oversold + Near lower BB + 4h HMA bullish
        elif crsi_oversold and near_lower_bb and hma_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: CRSI oversold + RSI < 40 + 4h HMA bullish
        elif crsi_oversold and rsi_oversold and hma_bullish:
            new_signal = SIZE_ENTRY
        # Path 4: Near lower BB + 4h HMA bullish + RSI < 45
        elif near_lower_bb and hma_bullish and rsi_14[i] < 45:
            new_signal = SIZE_ENTRY
        # Path 5: CRSI < 30 + Below BB SMA + 4h HMA bullish
        elif crsi[i] < 30 and below_bb_sma and hma_bullish:
            new_signal = SIZE_ENTRY
        # Path 6: Trending market + Pullback to BB SMA + 4h HMA bullish + RSI 35-50
        elif is_trending and below_bb_sma and hma_bullish and rsi_14[i] > 35 and rsi_14[i] < 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: CRSI overbought + 4h HMA bearish + Ranging market (mean reversion)
        if crsi_overbought and hma_bearish and is_ranging:
            new_signal = -SIZE_ENTRY
        # Path 2: CRSI overbought + Near upper BB + 4h HMA bearish
        elif crsi_overbought and near_upper_bb and hma_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: CRSI overbought + RSI > 60 + 4h HMA bearish
        elif crsi_overbought and rsi_overbought and hma_bearish:
            new_signal = -SIZE_ENTRY
        # Path 4: Near upper BB + 4h HMA bearish + RSI > 55
        elif near_upper_bb and hma_bearish and rsi_14[i] > 55:
            new_signal = -SIZE_ENTRY
        # Path 5: CRSI > 70 + Above BB SMA + 4h HMA bearish
        elif crsi[i] > 70 and above_bb_sma and hma_bearish:
            new_signal = -SIZE_ENTRY
        # Path 6: Trending market + Pullback to BB SMA + 4h HMA bearish + RSI 50-65
        elif is_trending and above_bb_sma and hma_bearish and rsi_14[i] > 50 and rsi_14[i] < 65:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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