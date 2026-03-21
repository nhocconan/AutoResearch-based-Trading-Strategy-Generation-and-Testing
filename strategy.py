#!/usr/bin/env python3
"""
Experiment #122: 30m HMA Trend + 4h HMA Filter + RSI Pullback + Volume
Hypothesis: 30m timeframe captures intraday trends while 4h HMA provides 
regime filter. RSI pullback entries (RSI<50 in uptrend, RSI>50 in downtrend)
ensure we're not chasing. Volume confirmation filters false breakouts.
Simpler conditions to ensure 10+ trades per symbol. Position sizing 0.25,
stoploss 2.5*ATR trailing. This adapts the current best (12h supertrend)
to 30m with HMA instead of Supertrend for faster response.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_4h_rsi_volume_v1"
timeframe = "30m"
leverage = 1.0

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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

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
    rsi = calculate_rsi(close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
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
        # 4h trend filter (HTF) - use completed 4h bar only
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # HMA cross signals (entry triggers)
        hma_cross_long = hma_21[i] > hma_50[i] and i > 0 and hma_21[i-1] <= hma_50[i-1]
        hma_cross_short = hma_21[i] < hma_50[i] and i > 0 and hma_21[i-1] >= hma_50[i-1]
        
        # RSI pullback (not chasing extremes)
        rsi_pullback_long = rsi[i] < 55 and rsi[i] > 30
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 70
        
        # Volume confirmation (avoid low volume fakeouts)
        volume_ok = vol_ratio[i] > 0.7
        
        # Price above/below HMA for confirmation
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + HMA trend + RSI pullback + volume
        if htf_bullish and hma_trend_long and rsi_pullback_long and volume_ok:
            new_signal = SIZE_ENTRY
        # HMA crossover entry (stronger signal)
        elif hma_cross_long and htf_bullish and volume_ok:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: 4h bearish + HMA trend + RSI pullback + volume
        if htf_bearish and hma_trend_short and rsi_pullback_short and volume_ok:
            new_signal = -SIZE_ENTRY
        # HMA crossover entry (stronger signal)
        elif hma_cross_short and htf_bearish and volume_ok:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic - trailing stop at 2.5*ATR
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
                new_signal = SIZE_EXIT
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
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
                trailing_stop = close[i] - 2.5 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.5 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.5 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.5 * atr[i]
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