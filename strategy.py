#!/usr/bin/env python3
"""
Experiment #095: 12h Fisher Transform + Daily HMA + Choppiness Regime Filter
Hypothesis: Fisher Transform catches reversals better than RSI in bear/range markets (2025 test).
Combine with Choppiness Index to detect regime: CHOP>61.8 = range (mean revert on Fisher extremes),
CHOP<38.2 = trend (follow Daily HMA direction). This adapts to market conditions instead of
always trend-following. Daily HMA provides HTF trend bias (proven in current best strategy).
Position sizing: 0.25 entry, 0.125 at 1.5R profit, stoploss at 2.5*ATR trailing.
12h timeframe balances trade frequency vs noise. Fisher period=9 for sensitivity.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_daily_hma_chop_regime_v1"
timeframe = "12h"
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
    Ehlers Fisher Transform - catches reversals in bear/range markets.
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 2 * (close - LL) / (HH - LL) - 1
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh == ll:
            x = 0.0
        else:
            x = 2.0 * (close[i] - ll) / (hh - ll) - 1.0
        
        # Clamp X to avoid log errors
        x = np.clip(x, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Trigger line is previous Fisher value
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - detects trend vs range regimes.
    CHOP > 61.8 = range/choppy (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh == ll or atr_sum == 0:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
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
    
    for i in range(100, n):
        # Daily trend filter (HTF) - price relative to Daily HMA
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Regime detection via Choppiness Index
        is_range = chop[i] > 61.8  # Mean reversion regime
        is_trend = chop[i] < 38.2  # Trend following regime
        
        # Fisher Transform signals
        fisher_cross_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_cross_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        fisher_extreme_long = fisher[i] < -2.0  # Oversold
        fisher_extreme_short = fisher[i] > 2.0  # Overbought
        
        # RSI filter
        rsi_ok_long = rsi[i] < 65
        rsi_ok_short = rsi[i] > 35
        
        new_signal = 0.0
        
        # LONG ENTRY conditions
        if is_range:
            # Mean reversion: Fisher extreme + RSI not too extreme
            if fisher_extreme_long and rsi_ok_long and daily_bullish:
                new_signal = SIZE_ENTRY
            # Fisher cross up from oversold
            elif fisher_cross_long and daily_bullish:
                new_signal = SIZE_ENTRY
        elif is_trend:
            # Trend following: Daily HMA bullish + price above HMA
            if daily_bullish and rsi_ok_long:
                new_signal = SIZE_ENTRY
        else:
            # Neutral regime: require stronger confirmation
            if fisher_cross_long and daily_bullish and rsi_ok_long:
                new_signal = SIZE_ENTRY
            elif fisher_extreme_long and daily_bullish:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        if is_range:
            # Mean reversion: Fisher extreme + RSI not too extreme
            if fisher_extreme_short and rsi_ok_short and daily_bearish:
                new_signal = -SIZE_ENTRY
            # Fisher cross down from overbought
            elif fisher_cross_short and daily_bearish:
                new_signal = -SIZE_ENTRY
        elif is_trend:
            # Trend following: Daily HMA bearish + price below HMA
            if daily_bearish and rsi_ok_short:
                new_signal = -SIZE_ENTRY
        else:
            # Neutral regime: require stronger confirmation
            if fisher_cross_short and daily_bearish and rsi_ok_short:
                new_signal = -SIZE_ENTRY
            elif fisher_extreme_short and daily_bearish:
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
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