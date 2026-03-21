#!/usr/bin/env python3
"""
Experiment #082: 4h Fisher Transform + Daily HMA Trend + Choppiness Regime
Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2025 test period). Combined with Daily HMA for trend bias and Choppiness Index
to detect regime (range vs trend). In range markets (CHOP>61.8), use Fisher reversals.
In trending markets (CHOP<38.2), follow Daily HMA direction. This regime-adaptive
approach should work better than pure trend-following which failed in 2022-2025.
Position sizing: 0.25 entry, 0.15 at 1.5R profit, stoploss at 2.5*ATR trailing.
4h timeframe provides good balance of signal quality and trade frequency.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_chop_regime_v2"
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
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Excellent for catching reversals in bear/range markets.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Calculate EMA of HL2
    ema_hl2 = hl2_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Calculate Fisher input (normalized price)
    # Clamp to avoid division by zero
    ema_hl2_arr = ema_hl2.values
    hl2_arr = hl2
    
    # Calculate range for normalization
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Normalize price to 0-1 range
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    normalized = (hl2_arr - lowest) / range_val
    
    # Clamp to avoid extreme values
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform: 0.5 * ln((1+x)/(1-x))
    fisher_input = 2 * normalized - 1
    fisher_input = np.clip(fisher_input, -0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + fisher_input) / (1 - fisher_input))
    
    # Signal line (EMA of Fisher)
    fisher_s = pd.Series(fisher)
    signal = fisher_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher, signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range-bound market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_s = pd.Series(tr)
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
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
        is_range = chop[i] > 55  # Range-bound market (slightly lower threshold for more trades)
        is_trend = chop[i] < 45  # Trending market
        
        # Fisher Transform signals
        fisher_cross_long = fisher[i] > -1.5 and (i > 0 and fisher[i-1] <= -1.5)
        fisher_cross_short = fisher[i] < 1.5 and (i > 0 and fisher[i-1] >= 1.5)
        
        # Fisher trend state
        fisher_bullish = fisher[i] > fisher_signal[i]
        fisher_bearish = fisher[i] < fisher_signal[i]
        
        # RSI filter
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        new_signal = 0.0
        
        # LONG ENTRY conditions
        # Condition 1: Range market + Fisher reversal long + RSI oversold
        if is_range and fisher_cross_long and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Condition 2: Trend market + Daily bullish + Fisher bullish
        elif is_trend and daily_bullish and fisher_bullish and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Condition 3: Fisher cross long + Daily bullish (regime-agnostic)
        elif fisher_cross_long and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: Range market + Fisher reversal short + RSI overbought
        if is_range and fisher_cross_short and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Condition 2: Trend market + Daily bearish + Fisher bearish
        elif is_trend and daily_bearish and fisher_bearish and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        # Condition 3: Fisher cross short + Daily bearish (regime-agnostic)
        elif fisher_cross_short and daily_bearish:
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
                profit = close[i] - entry_price
                risk = 2.5 * atr[i]
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
                profit = entry_price - close[i]
                risk = 2.5 * atr[i]
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