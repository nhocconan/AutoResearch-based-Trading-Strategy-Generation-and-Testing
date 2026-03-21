#!/usr/bin/env python3
"""
Experiment #294: 1d HMA Trend + Weekly Macro Bias + RSI Pullback with ATR Stops
Hypothesis: Daily timeframe captures major trend moves while weekly HMA provides macro bias.
RSI pullback entries (35-55 for long, 45-65 for short) ensure we enter on dips in uptrends and rallies in downtrends.
Simple entry logic ensures >=10 trades (learned from 0-trade failures in #282, #288).
ATR-based trailing stops (2.5*ATR) control drawdown. Position size 0.30 balances returns vs risk.
Target: Beat Sharpe=0.499 from current best while ensuring >=10 trades per symbol on 1d timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_weekly_bias_rsi_pullback_atr_v1"
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average for long-term trend filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
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
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Track previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_hma_21 = np.roll(hma_21, 1)
    prev_hma_21[0] = hma_21[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
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
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(sma_200[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        daily_bullish = close[i] > hma_21[i] and hma_21[i] > hma_50[i]
        daily_bearish = close[i] < hma_21[i] and hma_21[i] < hma_50[i]
        
        # Long-term trend filter (SMA200)
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # RSI pullback zones (generous ranges to ensure trades)
        rsi_pullback_long = 30 < rsi[i] < 55
        rsi_pullback_short = 45 < rsi[i] < 70
        rsi_not_extreme_long = rsi[i] < 75
        rsi_not_extreme_short = rsi[i] > 25
        
        # HMA crossover signals
        hma_cross_long = prev_close[i] <= prev_hma_21[i] and close[i] > hma_21[i]
        hma_cross_short = prev_close[i] >= prev_hma_21[i] and close[i] < hma_21[i]
        
        # HMA slope (trend direction)
        hma_slope_bullish = hma_21[i] > prev_hma_21[i]
        hma_slope_bearish = hma_21[i] < prev_hma_21[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Weekly bullish + Daily bullish + RSI pullback + HMA cross
        if weekly_bullish and daily_bullish and rsi_pullback_long and hma_cross_long:
            new_signal = SIZE_ENTRY
        # Secondary: Weekly bullish + Above SMA200 + RSI pullback + Price > HMA
        elif weekly_bullish and above_sma200 and rsi_pullback_long and close[i] > hma_21[i] and hma_slope_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: Daily bullish + RSI not extreme + HMA cross (simpler for more trades)
        elif daily_bullish and rsi_not_extreme_long and hma_cross_long:
            new_signal = SIZE_ENTRY
        # Quaternary: Price > HMA21 > HMA50 + RSI 40-60 (trend continuation)
        elif close[i] > hma_21[i] > hma_50[i] and 40 < rsi[i] < 60 and hma_slope_bullish:
            new_signal = SIZE_ENTRY
        # Simple: Weekly bullish + Price > HMA21 + RSI > 45
        elif weekly_bullish and close[i] > hma_21[i] and rsi[i] > 45 and rsi_not_extreme_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Weekly bearish + Daily bearish + RSI pullback + HMA cross
        if weekly_bearish and daily_bearish and rsi_pullback_short and hma_cross_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Weekly bearish + Below SMA200 + RSI pullback + Price < HMA
        elif weekly_bearish and below_sma200 and rsi_pullback_short and close[i] < hma_21[i] and hma_slope_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: Daily bearish + RSI not extreme + HMA cross (simpler for more trades)
        elif daily_bearish and rsi_not_extreme_short and hma_cross_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: Price < HMA21 < HMA50 + RSI 40-60 (trend continuation)
        elif close[i] < hma_21[i] < hma_50[i] and 40 < rsi[i] < 60 and hma_slope_bearish:
            new_signal = -SIZE_ENTRY
        # Simple: Weekly bearish + Price < HMA21 + RSI < 55
        elif weekly_bearish and close[i] < hma_21[i] and rsi[i] < 55 and rsi_not_extreme_short:
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