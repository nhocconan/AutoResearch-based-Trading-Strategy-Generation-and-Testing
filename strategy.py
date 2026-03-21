#!/usr/bin/env python3
"""
Experiment #069: 1h MACD Momentum + 4h HMA Trend + Volume Confirmation
Hypothesis: Recent 1h strategies failed due to over-filtering (RSI pullback too strict).
This uses simpler MACD histogram momentum with 4h HMA trend filter (proven in best strategy).
Add volume ratio confirmation (taker_buy_volume / volume > 0.55 for long) to filter fakeouts.
Looser entry conditions to ensure 10+ trades per symbol (learning from 0-trade failures).
Position sizing: 0.25 entry, stoploss at 2*ATR trailing. Discrete levels to minimize churn.
1h timeframe captures intraday trends while 4h HMA provides reliable HTF bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_macd_momentum_4h_hma_volume_v1"
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

def calculate_roc(close, period=10):
    """Calculate Rate of Change."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100
    return roc.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
    roc = calculate_roc(close, 10)
    
    # Volume ratio (taker buy pressure)
    volume_ratio = np.where(volume > 0, taker_buy_vol / volume, 0.5)
    
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
            hma_4h_val = close[i]  # fallback if aligned value is invalid
        
        trend_bullish = close[i] > hma_4h_val
        trend_bearish = close[i] < hma_4h_val
        
        # MACD momentum
        macd_bullish = histogram[i] > 0
        macd_bearish = histogram[i] < 0
        
        # MACD histogram increasing/decreasing
        hist_increasing = i > 0 and histogram[i] > histogram[i-1]
        hist_decreasing = i > 0 and histogram[i] < histogram[i-1]
        
        # ROC momentum confirmation
        roc_positive = roc[i] > 0
        roc_negative = roc[i] < 0
        
        # RSI filter (not extreme)
        rsi_ok_long = rsi[i] < 75
        rsi_ok_short = rsi[i] > 25
        
        # Volume confirmation
        volume_buy_pressure = volume_ratio[i] > 0.52
        volume_sell_pressure = volume_ratio[i] < 0.48
        
        new_signal = 0.0
        
        # LONG ENTRY - simpler conditions to ensure trades
        # Condition 1: 4h bullish + MACD bullish + histogram increasing
        if trend_bullish and macd_bullish and hist_increasing:
            new_signal = SIZE_ENTRY
        # Condition 2: 4h bullish + MACD cross up + volume confirmation
        elif trend_bullish and i > 0 and histogram[i-1] <= 0 and histogram[i] > 0 and volume_buy_pressure:
            new_signal = SIZE_ENTRY
        # Condition 3: 4h bullish + ROC positive + RSI ok + MACD bullish
        elif trend_bullish and roc_positive and rsi_ok_long and macd_bullish:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY
        # Condition 1: 4h bearish + MACD bearish + histogram decreasing
        if trend_bearish and macd_bearish and hist_decreasing:
            new_signal = -SIZE_ENTRY
        # Condition 2: 4h bearish + MACD cross down + volume confirmation
        elif trend_bearish and i > 0 and histogram[i-1] >= 0 and histogram[i] < 0 and volume_sell_pressure:
            new_signal = -SIZE_ENTRY
        # Condition 3: 4h bearish + ROC negative + RSI ok + MACD bearish
        elif trend_bearish and roc_negative and rsi_ok_short and macd_bearish:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest)
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
            
            # Calculate trailing stop (2*ATR from lowest)
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