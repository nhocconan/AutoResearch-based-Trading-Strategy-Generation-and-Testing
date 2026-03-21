#!/usr/bin/env python3
"""
Experiment #100: 4h RSI Pullback + MACD Momentum with Daily HMA Trend Filter
Hypothesis: Previous 4h strategies failed due to too many conflicting filters.
This uses simpler RSI pullback entries (RSI<40 in uptrend, RSI>60 in downtrend)
with MACD histogram confirmation. Daily HMA provides HTF trend bias (proven).
Fewer filters = more trades (critical for passing trade count requirements).
Position sizing: 0.30 entry, 0.15 at 1.5R profit, stoploss at 2.0*ATR trailing.
4h timeframe should generate 20-40 trades/year per symbol (enough for requirements).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_pullback_macd_daily_hma_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

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
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

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
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
    
    # EMA for trend
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # SMA for additional filter
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
    
    for i in range(200, n):
        # Daily trend filter (HTF) - price relative to Daily HMA
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h EMA trend
        ema_trend_long = ema_21[i] > ema_50[i]
        ema_trend_short = ema_21[i] < ema_50[i]
        
        # SMA 200 filter (long-term trend)
        sma_trend_long = close[i] > sma_200[i]
        sma_trend_short = close[i] < sma_200[i]
        
        # MACD signals
        macd_bullish = histogram[i] > 0
        macd_bearish = histogram[i] < 0
        macd_cross_long = histogram[i] > 0 and histogram[i-1] <= 0
        macd_cross_short = histogram[i] < 0 and histogram[i-1] >= 0
        
        # RSI pullback signals (simpler than before)
        # Long: RSI pulled back to 35-45 in uptrend
        rsi_pullback_long = 30 < rsi[i] < 50
        # Short: RSI rallied to 55-70 in downtrend
        rsi_pullback_short = 50 < rsi[i] < 70
        
        # MACD momentum confirmation
        macd_momentum_long = macd_bullish or macd_cross_long
        macd_momentum_short = macd_bearish or macd_cross_short
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (simpler to ensure trades)
        # Condition 1: RSI pullback + Daily bullish + EMA trend + MACD bullish
        if rsi_pullback_long and daily_bullish and ema_trend_long and macd_momentum_long:
            new_signal = SIZE_ENTRY
        # Condition 2: RSI pullback + SMA 200 bullish + MACD cross
        elif rsi_pullback_long and sma_trend_long and macd_cross_long:
            new_signal = SIZE_ENTRY
        # Condition 3: MACD cross + Daily bullish + RSI not overbought
        elif macd_cross_long and daily_bullish and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: RSI pullback + Daily bearish + EMA trend + MACD bearish
        if rsi_pullback_short and daily_bearish and ema_trend_short and macd_momentum_short:
            new_signal = -SIZE_ENTRY
        # Condition 2: RSI pullback + SMA 200 bearish + MACD cross
        elif rsi_pullback_short and sma_trend_short and macd_cross_short:
            new_signal = -SIZE_ENTRY
        # Condition 3: MACD cross + Daily bearish + RSI not oversold
        elif macd_cross_short and daily_bearish and rsi[i] > 35:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.0 * atr[i]
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