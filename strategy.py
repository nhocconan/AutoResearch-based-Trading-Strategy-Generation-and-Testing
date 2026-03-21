#!/usr/bin/env python3
"""
Experiment #145: 15m Multi-Timeframe Mean Reversion with 4h Trend Filter
Hypothesis: 15m timeframe is too noisy for pure trend following (see exp #133, #139 failures).
Instead, use 4h HMA for major trend direction, then enter on 15m RSI extremes (mean reversion)
when aligned with 4h trend. Add volume confirmation to avoid false breakouts.
Use Bollinger Band position for entry timing and ATR stoploss at 2.5*ATR.
This combines: HTF trend filter + LTF mean reversion + volume confirmation.
Position sizing: 0.25 entry, reduce to 0.12 at 2R profit, stoploss at 2.5*ATR.
Timeframe: 15m for more trade opportunities while 4h filter reduces whipsaws.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_rsi_mr_volume_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth, sma

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def calculate_kst(close, high, low, period1=10, period2=15, period3=20, period4=30, smooth1=10, smooth2=10, smooth3=10, smooth4=15):
    """
    Calculate Know Sure Thing (KST) momentum oscillator.
    Combines 4 ROC periods with different smoothing for multi-timeframe momentum.
    """
    roc1 = (close - np.roll(close, period1)) / np.roll(close, period1) * 100
    roc2 = (close - np.roll(close, period2)) / np.roll(close, period2) * 100
    roc3 = (close - np.roll(close, period3)) / np.roll(close, period3) * 100
    roc4 = (close - np.roll(close, period4)) / np.roll(close, period4) * 100
    
    roc1[0:period1] = 0
    roc2[0:period2] = 0
    roc3[0:period3] = 0
    roc4[0:period4] = 0
    
    sma1 = pd.Series(roc1).rolling(window=smooth1, min_periods=smooth1).mean().values
    sma2 = pd.Series(roc2).rolling(window=smooth2, min_periods=smooth2).mean().values
    sma3 = pd.Series(roc3).rolling(window=smooth3, min_periods=smooth3).mean().values
    sma4 = pd.Series(roc4).rolling(window=smooth4, min_periods=smooth4).mean().values
    
    kst = sma1 * 1 + sma2 * 2 + sma3 * 3 + sma4 * 4
    kst_signal = pd.Series(kst).rolling(window=9, min_periods=9).mean().values
    
    return kst, kst_signal

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
    bb_upper, bb_lower, bb_bw, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    vol_ratio = calculate_volume_ratio(volume, 20)
    kst, kst_signal = calculate_kst(close, high, low)
    
    # Calculate 15m HMA for local trend
    hma_15m_fast = calculate_hma(close, 8)
    hma_15m_slow = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major trend direction)
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m HMA crossover for local trend confirmation
        hma_cross_long = hma_15m_fast[i] > hma_15m_slow[i]
        hma_cross_short = hma_15m_fast[i] < hma_15m_slow[i]
        
        # RSI extremes for mean reversion entries
        rsi_oversold = rsi[i] < 32
        rsi_overbought = rsi[i] > 68
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # Bollinger Band position
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 0:
            bb_position = (close[i] - bb_lower[i]) / bb_range
        else:
            bb_position = 0.5
        bb_low = bb_position < 0.15
        bb_high = bb_position > 0.85
        
        # Volume confirmation (avoid low volume false breakouts)
        volume_confirmed = vol_ratio[i] > 0.8
        
        # KST momentum confirmation
        kst_bullish = kst[i] > kst_signal[i] if not np.isnan(kst_signal[i]) else False
        kst_bearish = kst[i] < kst_signal[i] if not np.isnan(kst_signal[i]) else False
        
        # KST divergence (momentum shifting)
        kst_rising = kst[i] > kst[i-3] if i > 3 else False
        kst_falling = kst[i] < kst[i-3] if i > 3 else False
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + RSI oversold + BB low + volume confirmed + KST turning up
        if trend_4h_bullish and rsi_oversold and bb_low and volume_confirmed:
            if kst_rising or kst_bullish:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: 4h bearish + RSI overbought + BB high + volume confirmed + KST turning down
        elif trend_4h_bearish and rsi_overbought and bb_high and volume_confirmed:
            if kst_falling or kst_bearish:
                new_signal = -SIZE_ENTRY
        
        # Alternative: HMA crossover with 4h trend alignment (trend following mode)
        if new_signal == 0.0:
            # LONG: 4h bullish + 15m HMA cross up + RSI not overbought
            if trend_4h_bullish and hma_cross_long and hma_15m_fast[i-1] <= hma_15m_slow[i-1] and rsi[i] < 60:
                if volume_confirmed:
                    new_signal = SIZE_ENTRY
            
            # SHORT: 4h bearish + 15m HMA cross down + RSI not oversold
            elif trend_4h_bearish and hma_cross_short and hma_15m_fast[i-1] >= hma_15m_slow[i-1] and rsi[i] > 40:
                if volume_confirmed:
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