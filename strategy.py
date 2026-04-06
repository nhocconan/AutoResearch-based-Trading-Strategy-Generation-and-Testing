#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h regime filter
# Elder Ray measures bull/bear power using EMA(13) as trend reference
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and Bear Power < 0 AND 12h EMA(50) rising (bull regime)
# Short when Bear Power > 0 and Bull Power < 0 AND 12h EMA(50) falling (bear regime)
# Uses volume confirmation to avoid false signals
# Target: 50-150 total trades over 4 years with balanced performance in bull/bear markets

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
    volume = prices['volume'].values
    
    # Elder Ray components - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume filter
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_filter = volume > vol_ma
    
    # 12h data for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_prev = np.roll(ema50_12h, 1)
    ema50_12h_prev[0] = ema50_12h[0]
    ema50_12h_rising = ema50_12h > ema50_12h_prev
    ema50_12h_falling = ema50_12h < ema50_12h_prev
    
    ema50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_rising)
    ema50_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema50_12h_rising_aligned[i]) or np.isnan(ema50_12h_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray turns bearish or regime changes
            elif bull_power[i] <= 0 or bear_power[i] >= 0 or not ema50_12h_rising_aligned[i]:
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
            # Exit: Elder Ray turns bullish or regime changes
            elif bear_power[i] <= 0 or bull_power[i] >= 0 or not ema50_12h_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and regime alignment
            if vol_filter[i]:
                # Long when Bull Power > 0, Bear Power < 0 AND bull regime
                if bull_power[i] > 0 and bear_power[i] < 0 and ema50_12h_rising_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when Bear Power > 0, Bull Power < 0 AND bear regime
                elif bear_power[i] > 0 and bull_power[i] < 0 and ema50_12h_falling_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals