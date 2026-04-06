#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 12h regime filter
# Bull Power (BP) = High - EMA13, Bear Power (BP) = Low - EMA13
# Long when Bull Power > 0 and Bear Power < 0 in bullish regime (ADX > 25 and +DI > -DI)
# Short when Bull Power < 0 and Bear Power > 0 in bearish regime (ADX > 25 and +DI < -DI)
# Uses 12h ADX for regime detection to avoid whipsaws in ranging markets
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets
# Uses 6h timeframe with 12h regime filter to reduce false signals

name = "6h_elderray_12h_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for regime filter (ADX) and EMA13 for Elder Ray
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # EMA13 for Elder Ray calculation
    ema13_12h = pd.Series(close_12h).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
        minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx, plus_di, minus_di
    
    adx_12h, plus_di_12h, minus_di_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # Align 12h data to 6h timeframe
    ema13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    plus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, plus_di_12h)
    minus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, minus_di_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema13_12h_aligned[i]) or np.isnan(bull_power_12h_aligned[i]) or 
            np.isnan(bear_power_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(plus_di_12h_aligned[i]) or np.isnan(minus_di_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray signals weaken or regime changes
            elif bull_power_12h_aligned[i] <= 0 or bear_power_12h_aligned[i] >= 0 or adx_12h_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray signals weaken or regime changes
            elif bull_power_12h_aligned[i] >= 0 or bear_power_12h_aligned[i] <= 0 or adx_12h_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with regime filter
            # Bullish regime: ADX > 25 and +DI > -DI
            # Bearish regime: ADX > 25 and +DI < -DI
            if adx_12h_aligned[i] > 25:
                if plus_di_12h_aligned[i] > minus_di_12h_aligned[i]:  # Bullish regime
                    # Long when Bull Power > 0 and Bear Power < 0
                    if bull_power_12h_aligned[i] > 0 and bear_power_12h_aligned[i] < 0:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                else:  # Bearish regime
                    # Short when Bull Power < 0 and Bear Power > 0
                    if bull_power_12h_aligned[i] < 0 and bear_power_12h_aligned[i] > 0:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals