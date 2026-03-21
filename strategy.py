#!/usr/bin/env python3
"""
Experiment #056: 30m RSI Mean Reversion with 4h HMA Trend Filter + Volume Confirmation
Hypothesis: 30m is too noisy for pure trend following (Supertrend failed). Instead,
use RSI mean reversion WITHIN the 4h trend direction. Enter long when RSI(7) dips
below 35 in 4h uptrend, enter short when RSI(7) rallies above 65 in 4h downtrend.
Add volume confirmation (volume > 1.5x 20-period avg) to filter false breakouts.
Use tighter 1.5*ATR stoploss for faster 30m timeframe. Position size 0.25-0.35.
This should generate more trades than 12h strategies while maintaining direction bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_4h_hma_volume_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=7):
    """Calculate RSI indicator with configurable period."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)  # Faster RSI for 30m
    vol_sma = calculate_volume_sma(volume, 20)
    
    # 30m HMA for local trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # 4h trend filter (HTF) - use HMA slope and price position
        hma_4h_valid = hma_4h_aligned[i] > 0
        fourh_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        fourh_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 30m HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Volume confirmation (spike above 1.5x average)
        volume_spike = vol_sma[i] > 0 and volume[i] > 1.5 * vol_sma[i]
        
        # RSI mean reversion signals (faster period=7 for 30m)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] >= 35 and rsi[i] <= 65
        
        # RSI momentum (rising/falling over last 3 bars)
        rsi_rising = (i > 2) and (rsi[i] > rsi[i-1]) and (rsi[i-1] > rsi[i-2])
        rsi_falling = (i > 2) and (rsi[i] < rsi[i-1]) and (rsi[i-1] < rsi[i-2])
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + RSI oversold + volume confirmation OR RSI turning up
        if fourh_bullish and rsi_oversold:
            if volume_spike or rsi_rising:
                new_signal = SIZE_ENTRY
            elif hma_trend_long:
                new_signal = SIZE_ENTRY
        elif fourh_bullish and hma_trend_long and rsi[i] < 45 and rsi_rising:
            new_signal = SIZE_ENTRY
        elif fourh_bullish and hma_trend_long and rsi[i] < 40:
            new_signal = SIZE_STRONG
        
        # SHORT ENTRY: 4h bearish + RSI overbought + volume confirmation OR RSI turning down
        if fourh_bearish and rsi_overbought:
            if volume_spike or rsi_falling:
                new_signal = -SIZE_ENTRY
            elif hma_trend_short:
                new_signal = -SIZE_ENTRY
        elif fourh_bearish and hma_trend_short and rsi[i] > 55 and rsi_falling:
            new_signal = -SIZE_ENTRY
        elif fourh_bearish and hma_trend_short and rsi[i] > 60:
            new_signal = -SIZE_STRONG
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop (1.5*ATR for faster 30m timeframe)
            current_stop = close[i] - 1.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 1.5 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 1.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 1.5 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 1.5 * atr[i] if position_side > 0 else close[i] + 1.5 * atr[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 1.5 * atr[i] if position_side > 0 else close[i] + 1.5 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals