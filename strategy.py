#!/usr/bin/env python3
"""
Experiment #644: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: 12h timeframe with regime-adaptive logic should handle both trending and 
choppy periods. Choppiness Index detects regime (CHOP>61.8=range, CHOP<38.2=trend).
Connors RSI excels at mean reversion in ranges (75% win rate). HMA provides HTF bias.

Key innovations:
1. Choppiness Index(14) regime detection - switch between mean revert and trend follow
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven mean reversion
3. 1d HMA(21) + 1w HMA(21) dual HTF bias - only trade with weekly trend
4. Regime-adaptive entries: CRSI<15 long in range, CRSI>85 short in range
5. Trend follow: HMA slope + price position in trending regime
6. ATR(14) trailing stop 2.5x for risk management
7. Discrete sizing: 0.25 base, 0.30 strong signals

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high - lowest_low < 1e-10:
            chop[i] = 100.0
            continue
        
        # Sum of ATR(1) over period
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10-15, Short when CRSI > 85-90
    """
    n = len(close)
    if n < rank_period + rsi_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.maximum(delta, 0)
    loss[1:] = np.maximum(-delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain[i] / avg_loss[i]))
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(streak_period + 1, n):
        streak = 0
        if close[i] > close[i-1]:
            streak = 1
            j = i - 1
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            streak = -1
            j = i - 1
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        
        # Convert streak to RSI-like value
        if streak >= 0:
            streak_rsi[i] = 100.0 * streak / (streak + 1) if streak > 0 else 50.0
        else:
            streak_rsi[i] = 100.0 * (abs(streak) + 1) / (abs(streak) + 2) if streak < 0 else 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and np.std(returns) > 1e-10:
            current_return = returns[-1] if len(returns) > 0 else 0
            rank = np.sum(returns[:-1] < current_return) / max(len(returns) - 1, 1)
            percent_rank[i] = rank * 100.0
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        in_range = chop[i] > 55.0  # Range/choppy market
        in_trend = chop[i] < 45.0  # Trending market
        
        # === HTF BIAS (1d + 1w HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === HMA SLOPE (12h) ===
        hma_slope_up = False
        hma_slope_down = False
        if i >= 3 and not np.isnan(hma_12h[i-3]):
            hma_slope_up = hma_12h[i] > hma_12h[i-1] and hma_12h[i-1] > hma_12h[i-2]
            hma_slope_down = hma_12h[i] < hma_12h[i-1] and hma_12h[i-1] < hma_12h[i-2]
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        desired_signal = 0.0
        
        if in_range:
            # MEAN REVERSION MODE (Connors RSI extremes)
            # Long when CRSI oversold + HTF not bearish
            if crsi[i] < 20.0 and not htf_bear:
                if crsi[i] < 15.0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Short when CRSI overbought + HTF not bullish
            elif crsi[i] > 80.0 and not htf_bull:
                if crsi[i] > 85.0:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        elif in_trend:
            # TREND FOLLOWING MODE (HMA slope + HTF alignment)
            # Long when HMA up + HTF bull + price above HMA
            if htf_bull and hma_slope_up and close[i] > hma_12h[i]:
                desired_signal = SIZE_STRONG
            elif hma_slope_up and close[i] > hma_12h[i] and not htf_bear:
                desired_signal = SIZE_BASE
            
            # Short when HMA down + HTF bear + price below HMA
            elif htf_bear and hma_slope_down and close[i] < hma_12h[i]:
                desired_signal = -SIZE_STRONG
            elif hma_slope_down and close[i] < hma_12h[i] and not htf_bull:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals