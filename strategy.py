#!/usr/bin/env python3
"""
Experiment #196: 4h KAMA Adaptive Trend + Fisher Transform Reversals + Volume Confirm
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than EMA/HMA,
reducing whipsaws in ranging markets (2022 crash bottom, 2025 bear). Fisher Transform catches
reversals at extremes with higher precision than RSI. Volume ratio (>1.5x 20-bar avg) confirms
breakouts are real. 1d HMA provides macro trend bias. This combination should work in both
bull and bear/range markets, unlike pure trend-following which failed in 2022/2025.
Position sizing: 0.25 entry, 0.125 half at 2R. Stoploss: 2.5*ATR trailing. Target: Beat Sharpe=0.499.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_volume_daily_hma_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average - adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER = trending (fast smoothing), Low ER = ranging (slow smoothing).
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio: net change / sum of absolute changes
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    volatility[0:period] = change[0:period]
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period-1] = close[period-1]
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extremes better than RSI.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price
        range_val = hh - ll
        if range_val > 0:
            value = 0.66 * ((close[i] - ll) / range_val - 0.5) + 0.67 * fisher[i-1] if i > period else 0
            value = np.clip(value, -0.999, 0.999)
        else:
            value = 0
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_avg
    ratio[vol_avg == 0] = 1.0
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # KAMA slope for trend direction
    kama_slope = kama - np.roll(kama, 5)
    kama_slope[0:5] = 0
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i] and kama_slope[i] > 0
        kama_bearish = close[i] < kama[i] and kama_slope[i] < 0
        
        # Fisher Transform signals (reversal detection)
        fisher_long = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]  # crossing up from oversold
        fisher_short = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]  # crossing down from overbought
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.3
        
        # KAMA pullback entry (price pulls back to KAMA in trend)
        kama_pullback_long = (close[i-1] < kama[i-1] and close[i] > kama[i] and 
                              kama_slope[i] > 0 and daily_bullish)
        kama_pullback_short = (close[i-1] > kama[i-1] and close[i] < kama[i] and 
                               kama_slope[i] < 0 and daily_bearish)
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Fisher reversal from oversold + volume + daily trend
        if fisher_long and volume_confirmed and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # KAMA pullback entry with volume
        elif kama_pullback_long and volume_confirmed:
            new_signal = SIZE_ENTRY
        
        # KAMA crossover with strong volume
        elif close[i] > kama[i] and close[i-1] <= kama[i-1] and volume_confirmed and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Fisher reversal from overbought + volume + daily trend
        if fisher_short and volume_confirmed and daily_bearish:
            new_signal = -SIZE_ENTRY
        
        # KAMA pullback entry with volume
        elif kama_pullback_short and volume_confirmed:
            new_signal = -SIZE_ENTRY
        
        # KAMA crossover with strong volume
        elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume_confirmed and daily_bearish:
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