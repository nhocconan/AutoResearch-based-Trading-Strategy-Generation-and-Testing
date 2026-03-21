#!/usr/bin/env python3
"""
Experiment #096: 1d RSI Mean Reversion with Weekly HMA Trend Filter
Hypothesis: Daily timeframe needs simpler entry logic to ensure 10+ trades.
Use Weekly HMA for trend bias (proven in best strategy), RSI(14) for mean reversion
entries within trend. Long when RSI<40 + price>Weekly_HMA, Short when RSI>60 + price<Weekly_HMA.
Add ATR(14) volatility filter to avoid low-vol chop. Position size 0.25, stoploss 2.5*ATR.
This should work in both bull (2021) and bear (2022, 2025) markets by following HTF trend.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_weekly_hma_mr_v1"
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

def calculate_volatility_ratio(close, period=20):
    """Calculate volatility ratio (current ATR / average ATR)."""
    returns = np.diff(close, prepend=close[0]) / close
    abs_returns = np.abs(returns)
    avg_vol = pd.Series(abs_returns).rolling(window=period, min_periods=period).mean().values
    current_vol = pd.Series(abs_returns).rolling(window=5, min_periods=5).mean().values
    vol_ratio = current_vol / (avg_vol + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volatility_ratio(close, 20)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(250, n):  # Need enough data for 200 SMA + weekly alignment
        # Weekly trend filter (HTF) - price relative to Weekly HMA
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend confirmation
        daily_bullish = close[i] > sma_50[i] and sma_50[i] > sma_200[i]
        daily_bearish = close[i] < sma_50[i] and sma_50[i] < sma_200[i]
        
        # RSI mean reversion signals (moderate thresholds for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_extreme_long = rsi[i] < 35
        rsi_extreme_short = rsi[i] > 65
        
        # Volatility filter (avoid very low vol)
        vol_ok = vol_ratio[i] > 0.7
        
        # Volume confirmation
        vol_confirm = volume[i] > 0.8 * vol_sma[i]
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (simpler for more trades)
        # Primary: Weekly bullish + RSI oversold + volatility OK
        if weekly_bullish and rsi_oversold and vol_ok:
            new_signal = SIZE_ENTRY
        # Secondary: Weekly bullish + RSI extreme + volume confirm
        elif weekly_bullish and rsi_extreme_long and vol_confirm:
            new_signal = SIZE_ENTRY
        # Tertiary: Daily bullish + RSI oversold (catch trend continuations)
        elif daily_bullish and rsi_oversold and vol_ok:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Primary: Weekly bearish + RSI overbought + volatility OK
        if weekly_bearish and rsi_overbought and vol_ok:
            new_signal = -SIZE_ENTRY
        # Secondary: Weekly bearish + RSI extreme + volume confirm
        elif weekly_bearish and rsi_extreme_short and vol_confirm:
            new_signal = -SIZE_ENTRY
        # Tertiary: Daily bearish + RSI overbought (catch trend continuations)
        elif daily_bearish and rsi_overbought and vol_ok:
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
                profit = close[i] - entry_price
                risk = 2.5 * atr[i]
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
                profit = entry_price - close[i]
                risk = 2.5 * atr[i]
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