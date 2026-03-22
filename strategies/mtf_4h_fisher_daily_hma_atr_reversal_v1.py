#!/usr/bin/env python3
"""
Experiment #154: 4h Fisher Transform Reversals with Daily HMA Trend Filter
Hypothesis: Fisher Transform excels at identifying turning points in bear/range markets
(2022 crash, 2025 consolidation). Unlike RSI which can stay oversold/overbought for
extended periods, Fisher Transform normalizes price to Gaussian distribution, creating
clearer reversal signals at extremes (-2.0/+2.0). Daily HMA provides major trend bias
to avoid counter-trend traps. Entry thresholds loosened (Fisher<-1.5 or >1.5) to ensure
sufficient trades. ATR stoploss at 2.5*ATR protects capital. Position sizing: 0.30 entry,
0.15 at 2R profit, discrete levels minimize fee churn. This targets the weakness of
pure trend-following strategies that failed in 2022-2025 bear/range conditions.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_atr_reversal_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer reversal signals.
    Reference: Ehlers, J.F. (2002) "Fisher Transform"
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Calculate EMA of hl2
    ema_hl2 = hl2_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Normalize price (0 to 1 range)
    min_hl2 = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    max_hl2 = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    
    range_hl2 = max_hl2 - min_hl2
    range_hl2 = np.where(range_hl2 > 0, range_hl2, 1e-10)
    
    normalized = (hl2 - min_hl2) / range_hl2
    normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher_input = np.log(normalized / (1 - normalized))
    fisher = pd.Series(fisher_input).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Fisher trigger line (1-period lag)
    fisher_trigger = np.roll(fisher, 1)
    fisher_trigger[0] = fisher[0]
    
    return fisher, fisher_trigger

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    sma_200 = calculate_sma(close, 200)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
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
    
    for i in range(100, n):
        # Daily trend filter (major trend direction)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 4h trend filter
        trend_bullish = hma_20[i] > hma_50[i]
        trend_bearish = hma_20[i] < hma_50[i]
        
        # Price position relative to SMA200
        above_sma200 = sma_200[i] > 0 and close[i] > sma_200[i]
        below_sma200 = sma_200[i] > 0 and close[i] < sma_200[i]
        
        # Fisher Transform reversal signals (loosened for more trades)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher_trigger[i] < -1.0
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher_trigger[i] > 1.0
        
        # RSI confirmation (wider thresholds for more trades)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else False
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else False
        
        new_signal = 0.0
        
        # LONG ENTRY: Fisher oversold + RSI confirmation + Daily not strongly bearish
        if fisher_oversold or fisher_cross_up:
            if daily_bullish and rsi_oversold:
                # Strong long: daily bullish + oversold
                new_signal = SIZE_ENTRY
            elif not daily_bearish and rsi_rising:
                # Moderate long: daily neutral + RSI rising
                new_signal = SIZE_ENTRY
            elif above_sma200 and fisher_cross_up:
                # Above SMA200 + Fisher cross up
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Fisher overbought + RSI confirmation + Daily not strongly bullish
        elif fisher_overbought or fisher_cross_down:
            if daily_bearish and rsi_overbought:
                # Strong short: daily bearish + overbought
                new_signal = -SIZE_ENTRY
            elif not daily_bullish and rsi_falling:
                # Moderate short: daily neutral + RSI falling
                new_signal = -SIZE_ENTRY
            elif below_sma200 and fisher_cross_down:
                # Below SMA200 + Fisher cross down
                new_signal = -SIZE_ENTRY
        
        # TREND FOLLOWING: HMA crossover with Fisher confirmation
        if new_signal == 0.0:
            if trend_bullish and hma_20[i-1] <= hma_50[i-1] and fisher[i] > -1.0:
                if daily_bullish or above_sma200:
                    new_signal = SIZE_ENTRY
            
            elif trend_bearish and hma_20[i-1] >= hma_50[i-1] and fisher[i] < 1.0:
                if daily_bearish or below_sma200:
                    new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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