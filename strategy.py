#!/usr/bin/env python3
"""
Experiment #152: 30m Trend-Following with 4h HMA Filter and Donchian Breakout
Hypothesis: 30m timeframe needs stronger trend filter to avoid whipsaw. Using 4h HMA
for major trend direction, 30m Donchian(20) breakout for entry trigger, RSI(14) to
avoid entering at extremes, and volume confirmation. This combines proven elements
from best strategies (4h HMA trend filter) with breakout mechanics that work in
both bull and bear markets. Conservative position sizing (0.25) with 2.5*ATR stop
controls drawdown. Fewer but higher quality trades target 30-50 per year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_4h_hma_donchian_rsi_vol_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = 0.0
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    volatility[0] = change[0]
    er = np.where(volatility > 0, change / volatility, 0.0)
    
    # Calculate smoothing constant
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

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
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    kama = calculate_kama(close, 10)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
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
        # 4h trend filter (major trend direction)
        trend_4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        
        # 30m trend filter
        trend_30m_bullish = hma_20[i] > hma_50[i] and close[i] > hma_20[i]
        trend_30m_bearish = hma_20[i] < hma_50[i] and close[i] < hma_20[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i-1] and donchian_upper[i] > 0
        donchian_breakout_short = close[i] < donchian_lower[i-1] and donchian_lower[i] > 0
        
        # RSI filter (avoid extremes for trend following)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        rsi_momentum_long = rsi[i] > 50 and rsi[i] > rsi[i-3] if i > 3 else False
        rsi_momentum_short = rsi[i] < 50 and rsi[i] < rsi[i-3] if i > 3 else False
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_sma[i] if vol_sma[i] > 0 else False
        
        # ATR volatility filter (avoid low vol chop)
        atr_ratio = atr[i] / close[i] * 100 if close[i] > 0 else 0
        vol_ok = atr_ratio > 0.5  # At least 0.5% ATR
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + Donchian breakout + RSI momentum + Volume
        if donchian_breakout_long and trend_4h_bullish:
            if (trend_30m_bullish or kama_bullish) and rsi_not_overbought and rsi_momentum_long:
                if volume_confirmed and vol_ok:
                    new_signal = SIZE_ENTRY
                elif not volume_confirmed and rsi[i] > 55:
                    new_signal = SIZE_ENTRY * 0.8
        
        # SHORT ENTRY: 4h bearish + Donchian breakdown + RSI momentum + Volume
        elif donchian_breakout_short and trend_4h_bearish:
            if (trend_30m_bearish or kama_bearish) and rsi_not_oversold and rsi_momentum_short:
                if volume_confirmed and vol_ok:
                    new_signal = -SIZE_ENTRY
                elif not volume_confirmed and rsi[i] < 45:
                    new_signal = -SIZE_ENTRY * 0.8
        
        # PULLBACK ENTRY: Trend established, RSI pullback
        elif trend_4h_bullish and trend_30m_bullish:
            if rsi[i] < 45 and rsi[i] > rsi[i-2] if i > 2 else False:
                if close[i] > hma_4h_aligned[i] and vol_ok:
                    new_signal = SIZE_ENTRY * 0.6
        
        elif trend_4h_bearish and trend_30m_bearish:
            if rsi[i] > 55 and rsi[i] < rsi[i-2] if i > 2 else False:
                if close[i] < hma_4h_aligned[i] and vol_ok:
                    new_signal = -SIZE_ENTRY * 0.6
        
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