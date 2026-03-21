#!/usr/bin/env python3
"""
Experiment #300: 1d HMA Trend + Weekly Bias + Volume Confirmation with ATR Stops
Hypothesis: Daily timeframe with weekly macro bias works, but #294 had too many AND conditions causing few trades.
This version simplifies entry logic: HMA crossover + RSI zone + volume confirmation (above 20-day avg).
Fewer filters = more trades while maintaining quality. Weekly HMA provides macro trend direction.
ATR trailing stop (2.5*ATR) controls drawdown. Position size 0.30 balances returns vs risk.
Target: Beat Sharpe=0.499 while ensuring >=10 trades per symbol on 1d timeframe.
Key change: Remove SMA200 filter, simplify RSI ranges, add volume confirmation for breakout validity.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_weekly_volume_rsi_atr_v1"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
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

def calculate_volume_sma(volume, period=20):
    """Calculate volume moving average for volume confirmation."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

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
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Track previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_hma_21 = np.roll(hma_21, 1)
    prev_hma_21[0] = hma_21[0]
    prev_hma_50 = np.roll(hma_50, 1)
    prev_hma_50[0] = hma_50[0]
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias (simpler: just price vs weekly HMA)
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Daily trend filter (HMA21 vs HMA50)
        daily_bullish = hma_21[i] > hma_50[i]
        daily_bearish = hma_21[i] < hma_50[i]
        
        # Volume confirmation (above average)
        volume_confirmed = volume[i] > vol_sma[i]
        
        # RSI zones (generous ranges to ensure trades)
        rsi_long_ok = 35 < rsi[i] < 65
        rsi_short_ok = 35 < rsi[i] < 65
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # HMA crossover signals
        hma_cross_long = prev_close[i] <= prev_hma_21[i] and close[i] > hma_21[i]
        hma_cross_short = prev_close[i] >= prev_hma_21[i] and close[i] < hma_21[i]
        
        # HMA slope (trend direction)
        hma_slope_bullish = hma_21[i] > prev_hma_21[i]
        hma_slope_bearish = hma_21[i] < prev_hma_21[i]
        
        # HMA50 slope
        hma50_slope_bullish = hma_50[i] > prev_hma_50[i]
        hma50_slope_bearish = hma_50[i] < prev_hma_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Weekly bullish + Daily bullish + HMA cross + Volume + RSI ok
        if weekly_bullish and daily_bullish and hma_cross_long and volume_confirmed and rsi_long_ok:
            new_signal = SIZE_ENTRY
        # Secondary: Weekly bullish + Price > HMA21 > HMA50 + Volume + RSI not overbought
        elif weekly_bullish and close[i] > hma_21[i] > hma_50[i] and volume_confirmed and rsi_not_overbought and hma_slope_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: Daily bullish + HMA cross + Volume (simpler for more trades)
        elif daily_bullish and hma_cross_long and volume_confirmed and rsi_not_overbought:
            new_signal = SIZE_ENTRY
        # Quaternary: Price > HMA21 + HMA21 > HMA50 + RSI 40-60 + volume (trend continuation)
        elif close[i] > hma_21[i] > hma_50[i] and 40 < rsi[i] < 60 and volume_confirmed and hma50_slope_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Weekly bearish + Daily bearish + HMA cross + Volume + RSI ok
        if weekly_bearish and daily_bearish and hma_cross_short and volume_confirmed and rsi_short_ok:
            new_signal = -SIZE_ENTRY
        # Secondary: Weekly bearish + Price < HMA21 < HMA50 + Volume + RSI not oversold
        elif weekly_bearish and close[i] < hma_21[i] < hma_50[i] and volume_confirmed and rsi_not_oversold and hma_slope_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: Daily bearish + HMA cross + Volume (simpler for more trades)
        elif daily_bearish and hma_cross_short and volume_confirmed and rsi_not_oversold:
            new_signal = -SIZE_ENTRY
        # Quaternary: Price < HMA21 < HMA50 + RSI 40-60 + volume (trend continuation)
        elif close[i] < hma_21[i] < hma_50[i] and 40 < rsi[i] < 60 and volume_confirmed and hma50_slope_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
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