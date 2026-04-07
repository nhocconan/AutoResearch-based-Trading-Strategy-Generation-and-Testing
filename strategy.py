#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour 123 Reversal pattern with 1d trend filter and volume confirmation
# The 123 Reversal is a price action pattern where: 1) price makes a swing high/low, 
# 2) pulls back to form a swing point 2, 3) breaks through swing point 1 with momentum.
# In strong trends, we look for breakouts in trend direction; in weak trends, we look for reversals.
# Uses 1d EMA50 for trend filter and volume > 1.3x average for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) with precise entry conditions.

name = "6h_123_reversal_1d_ema50_vol_v1"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 6h swing points for 123 pattern
    # Swing high: higher high followed by lower high
    # Swing low: lower low followed by higher low
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    
    for i in range(2, n-2):
        # Swing high: high[i] > high[i-1] and high[i] > high[i+1] and high[i-1] > high[i-2] and high[i+1] > high[i+2]
        if (high[i] > high[i-1] and high[i] > high[i+1] and 
            high[i-1] > high[i-2] and high[i+1] > high[i+2]):
            swing_high[i] = high[i]
        # Swing low: low[i] < low[i-1] and low[i] < low[i+1] and low[i-1] < low[i-2] and low[i+1] < low[i+2]
        if (low[i] < low[i-1] and low[i] < low[i+1] and 
            low[i-1] < low[i-2] and low[i+1] < low[i+2]):
            swing_low[i] = low[i]
    
    # Forward fill swing points to use as reference levels
    swing_high_ff = pd.Series(swing_high).replace(0, np.nan).ffill().values
    swing_low_ff = pd.Series(swing_low).replace(0, np.nan).ffill().values
    
    # 6h volume average for confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(swing_high_ff[i]) or np.isnan(swing_low_ff[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_6h_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below swing low (failed breakout) or reaches 2x risk
            elif close[i] < swing_low_ff[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above swing high (failed breakdown) or reaches 2x risk
            elif close[i] > swing_high_ff[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for 123 pattern entries
            # Uptrend (price > EMA50): look for 123 breakout longs
            # Downtrend (price < EMA50): look for 123 breakdown shorts
            
            # Long 123 breakout: 
            # 1. Swing low forms (point 1)
            # 2. Pullback to form higher low (point 2) 
            # 3. Break above point 1 with volume (point 3)
            point1_low = swing_low_ff[i]
            if not np.isnan(point1_low) and point1_low > 0:
                # Check for higher low formation (point 2) and breakout
                if (low[i] > point1_low and  # higher low (point 2 forming)
                    close[i] > point1_low and  # breaks point 1 (point 3)
                    close[i] > ema_50_aligned[i] and  # in uptrend
                    volume[i] > 1.3 * volume_ma_6h_aligned[i]):  # volume confirmation
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            
            # Short 123 breakdown:
            # 1. Swing high forms (point 1)
            # 2. Pullback to form lower high (point 2)
            # 3. Break below point 1 with volume (point 3)
            point1_high = swing_high_ff[i]
            if not np.isnan(point1_high) and point1_high > 0:
                # Check for lower high formation (point 2) and breakdown
                if (high[i] < point1_high and  # lower high (point 2 forming)
                    close[i] < point1_high and  # breaks point 1 (point 3)
                    close[i] < ema_50_aligned[i] and  # in downtrend
                    volume[i] > 1.3 * volume_ma_6h_aligned[i]):  # volume confirmation
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals