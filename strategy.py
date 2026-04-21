#3e
#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator crossover with 1w EMA50 trend filter and volume confirmation.
Long when Alligator Lips (3) crosses above Jaw (13) with price > EMA50 and volume > 1.5x average.
Short when Lips crosses below Jaw with price < EMA50 and volume > 1.5x average.
Exit on opposite crossover or 2x ATR stop.
Designed for 10-25 trades/year to minimize fee decay while capturing trend changes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Williams Alligator components (SMMA = smoothed moving average)
    close = prices['close'].values
    # SMMA formula: today = (yesterday * (period-1) + today) / period
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # Initialize with SMA
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Volume confirmation: volume spike > 1.5x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        lips_val = lips[i]
        jaw_val = jaw[i]
        ema_50_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: Lips crosses above Jaw with price > EMA50 and volume
            if (lips_val > jaw_val and 
                lips[i-1] <= jaw[i-1] and  # crossover confirmation
                price_close > ema_50_val and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips crosses below Jaw with price < EMA50 and volume
            elif (lips_val < jaw_val and 
                  lips[i-1] >= jaw[i-1] and  # crossover confirmation
                  price_close < ema_50_val and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite crossover or ATR-based stoploss
            exit_signal = False
            
            # Opposite crossover exit
            if position == 1 and lips_val < jaw_val and lips[i-1] >= jaw[i-1]:
                exit_signal = True
            elif position == -1 and lips_val > jaw_val and lips[i-1] <= jaw[i-1]:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from recent extreme)
            if position == 1:
                # For longs, stop below recent low minus 2*ATR
                if price_low < np.min(prices['low'].iloc[max(0, i-5):i+1]) - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above recent high plus 2*ATR
                if price_high > np.max(prices['high'].iloc[max(0, i-5):i+1]) + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_LipsJawCross_1wEMA50_Trend_Volume1.5x_ATR2x"
timeframe = "1d"
leverage = 1.0