#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla Pivot Breakout with 4h EMA200 trend filter and volume spike.
Long when price breaks above R1 AND close > 4h EMA200 AND volume > 2x average.
Short when price breaks below S1 AND close < 4h EMA200 AND volume > 2x average.
Exit when price reverts to pivot point (PP) or volume drops below average.
Camarilla pivots provide intraday support/resistance levels that work well in ranging markets.
4h EMA200 filters for higher timeframe trend direction to avoid counter-trend trades.
Volume spike confirms conviction behind the breakout.
Designed for 1h timeframe targeting 80-120 total trades over 4 years with moderate frequency.
Works in both bull and bear markets by only taking trend-aligned breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivots for 1h timeframe using previous bar's OHLC
    # Camarilla levels: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    
    # Load 4h data for EMA200 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA200 on 4h data
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMA200 to 1h timeframe
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Volume average (24-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pp[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema200_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_val = ema200_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 4h EMA200 AND volume spike
            if (price > r1[i] and price > ema200_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND price < 4h EMA200 AND volume spike
            elif (price < s1[i] and price < ema200_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot point OR volume drops below average
                if (price <= pp[i] or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot point OR volume drops below average
                if (price >= pp[i] or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1_S1_Breakout_4hEMA200_Volume"
timeframe = "1h"
leverage = 1.0