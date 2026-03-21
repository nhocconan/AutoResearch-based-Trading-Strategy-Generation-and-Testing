#!/usr/bin/env python3
"""
Experiment #079: 15m EMA Crossover with 4h HMA Trend + Volume Filter
Hypothesis: 15m timeframe needs simpler entry logic to ensure 10+ trades per symbol.
Previous 15m attempts failed due to overly strict filters (RSI pullback + ADX).
This uses: 4h HMA for trend bias (proven HTF filter), 15m EMA(12/26) crossover for entries,
RSI(14) moderate filter (not extreme), volume > 20-bar average for confirmation.
Position sizing: 0.25 entry, stoploss at 2.0*ATR trailing. Discrete levels to minimize churn.
Key: Fewer conflicting filters = more trades while maintaining HTF trend alignment.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_ema_crossover_4h_hma_volume_v1"
timeframe = "15m"
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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # EMA for crossover
    ema_12 = calculate_ema(close, 12)
    ema_26 = calculate_ema(close, 26)
    
    # Volume confirmation
    vol_sma = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF) - price relative to 4h HMA
        hma_4h_val = hma_4h_aligned[i]
        if np.isnan(hma_4h_val) or hma_4h_val == 0:
            hma_4h_val = close[i]  # fallback if alignment fails
        
        daily_bullish = close[i] > hma_4h_val
        daily_bearish = close[i] < hma_4h_val
        
        # 15m EMA crossover
        ema_cross_long = ema_12[i] > ema_26[i] and (i > 0 and ema_12[i-1] <= ema_26[i-1])
        ema_cross_short = ema_12[i] < ema_26[i] and (i > 0 and ema_12[i-1] >= ema_26[i-1])
        
        # EMA trend state (sustained trend, not just cross)
        ema_trend_long = ema_12[i] > ema_26[i]
        ema_trend_short = ema_12[i] < ema_26[i]
        
        # RSI filter (moderate, not extreme - ensures more trades)
        rsi_ok_long = rsi[i] < 65  # Not overbought
        rsi_ok_short = rsi[i] > 35  # Not oversold
        
        # Volume confirmation (above average)
        vol_ok = volume[i] > vol_sma[i] if not np.isnan(vol_sma[i]) else True
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (simpler to ensure 10+ trades)
        # Primary: EMA cross + 4h bullish + RSI ok
        if ema_cross_long and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: EMA trend + 4h bullish + volume confirmation
        elif ema_trend_long and daily_bullish and vol_ok and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Primary: EMA cross + 4h bearish + RSI ok
        if ema_cross_short and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: EMA trend + 4h bearish + volume confirmation
        elif ema_trend_short and daily_bearish and vol_ok and rsi_ok_short:
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
                new_signal = SIZE_EXIT
        
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
                new_signal = SIZE_EXIT
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.0 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.0 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.0 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.0 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals