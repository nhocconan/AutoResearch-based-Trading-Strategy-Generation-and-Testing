#!/usr/bin/env python3
"""
Experiment #489: 1h Connors RSI + Choppiness Regime + 4h HMA Bias + ATR Stop
Hypothesis: 1h timeframe needs mean-reversion approach with regime filter.
Connors RSI (CRSI) catches oversold/overbought extremes with 75% win rate.
Choppiness Index filters: CHOP>61.8 = range (use mean reversion), CHOP<38.2 = trend (reduce entries).
4h HMA provides HTF trend bias to avoid counter-trend mean reversion.
Multiple entry paths ensure >=10 trades. Conservative sizing (0.25) controls DD.
2.0*ATR stoploss for 1h bars. Must beat Sharpe=0.499 baseline.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_4h_hma_regime_atr_v1"
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
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Catches extreme oversold/overbought with high win rate.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_rsi = calculate_rsi(np.abs(streak), streak_period)
    # Invert for down streaks
    streak_rsi = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank: rank of today's change vs last 100 days
    for i in range(rank_period, n):
        changes = np.diff(close[i-rank_period:i+1])
        today_change = changes[-1]
        rank = np.sum(changes[:-1] <= today_change) / (len(changes) - 1) * 100
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures if market is trending or ranging.
    CHOP > 61.8 = range (mean reversion works)
    CHOP < 38.2 = trend (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 50.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    hma_21 = calculate_hma(close, 21)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h trend
        hma_21_bullish = close[i] > hma_21[i]
        hma_21_bearish = close[i] < hma_21[i]
        
        # SMA200 filter
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # Choppiness regime
        is_range = chop[i] > 55.0  # Slightly lowered from 61.8 for more trades
        is_trend = chop[i] < 45.0  # Slightly raised from 38.2
        
        # CRSI extremes
        crsi_oversold = crsi[i] < 20.0  # Loosened from 10 for more trades
        crsi_overbought = crsi[i] > 80.0  # Loosened from 90
        
        # CRSI recovery
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # RSI confirmation
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: CRSI oversold + Range regime + Above SMA200 + 4h bullish
        if crsi_oversold and is_range and above_sma200 and hma_4h_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: CRSI oversold + CRSI rising + RSI oversold
        elif crsi_oversold and crsi_rising and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 3: CRSI very oversold (<15) + 4h not bearish
        elif crsi[i] < 15.0 and not hma_4h_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 4: Range regime + RSI oversold + Price > HMA21 (pullback in range)
        elif is_range and rsi_oversold and hma_21_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 5: CRSI < 25 + RSI < 40 + Above SMA200
        elif crsi[i] < 25.0 and rsi_14[i] < 40.0 and above_sma200:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: CRSI overbought + Range regime + Below SMA200 + 4h bearish
        if crsi_overbought and is_range and below_sma200 and hma_4h_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: CRSI overbought + CRSI falling + RSI overbought
        elif crsi_overbought and crsi_falling and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 3: CRSI very overbought (>85) + 4h not bullish
        elif crsi[i] > 85.0 and not hma_4h_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Range regime + RSI overbought + Price < HMA21 (rally in range)
        elif is_range and rsi_overbought and hma_21_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 5: CRSI > 75 + RSI > 60 + Below SMA200
        elif crsi[i] > 75.0 and rsi_14[i] > 60.0 and below_sma200:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 1h timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 1h timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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