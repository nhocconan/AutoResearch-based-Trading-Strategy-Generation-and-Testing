#!/usr/bin/env python3
"""
Experiment #114: 1d Supertrend + Weekly HMA Trend + RSI Pullback
Hypothesis: Daily timeframe with Weekly HTF trend filter should reduce whipsaws
compared to 12h strategies. Supertrend provides clear trend direction, Weekly HMA
filters major trend bias, RSI pullback ensures we enter on dips in uptrends.
Position sizing: 0.30 entry, 0.15 at 1.5R profit, stoploss at 2.5*ATR trailing.
This should generate 15-30 trades/year on daily bars while maintaining quality.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_weekly_hma_rsi_pullback_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    trend = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    trend[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            trend[i] = 1
        else:
            supertrend[i] = upper_band[i]
            trend[i] = -1
    
    return supertrend, trend

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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    sma_200 = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):  # Start after 200 SMA + buffer
        # Weekly trend filter (HTF) - price relative to Weekly HMA
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filter - price above/below 200 SMA
        daily_bullish = close[i] > sma_200[i]
        daily_bearish = close[i] < sma_200[i]
        
        # Supertrend signals
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_trend[i] == 1 and (i > 0 and st_trend[i-1] == -1)
        st_flip_short = st_trend[i] == -1 and (i > 0 and st_trend[i-1] == 1)
        
        # RSI pullback filter (enter on dips in uptrend, rallies in downtrend)
        rsi_pullback_long = rsi[i] < 55 and rsi[i] > 30  # Pullback but not oversold
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 70  # Rally but not overbought
        
        # RSI extreme for contrarian entries
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (multiple pathways to ensure trades)
        # Condition 1: Supertrend flip + Weekly bullish + Daily bullish
        if st_flip_long and weekly_bullish and daily_bullish:
            new_signal = SIZE_ENTRY
        # Condition 2: Supertrend bullish + Weekly bullish + RSI pullback
        elif st_bullish and weekly_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Condition 3: Supertrend flip + RSI oversold (contrarian bounce)
        elif st_flip_long and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Condition 4: Price above both Weekly HMA and 200 SMA + Supertrend bullish
        elif weekly_bullish and daily_bullish and st_bullish and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: Supertrend flip + Weekly bearish + Daily bearish
        if st_flip_short and weekly_bearish and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Condition 2: Supertrend bearish + Weekly bearish + RSI pullback
        elif st_bearish and weekly_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Condition 3: Supertrend flip + RSI overbought (contrarian drop)
        elif st_flip_short and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Condition 4: Price below both Weekly HMA and 200 SMA + Supertrend bearish
        elif weekly_bearish and daily_bearish and st_bearish and rsi[i] > 40:
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