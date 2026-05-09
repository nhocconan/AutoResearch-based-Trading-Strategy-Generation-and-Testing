#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Keltner_Reversal_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h Keltner reversal with daily trend filter and volume spike.
    - Long: Close below lower Keltner band (20, 1.5) + daily close > daily EMA50 + volume > 2x avg
    - Short: Close above upper Keltner band (20, 1.5) + daily close < daily EMA50 + volume > 2x avg
    - Exit: Close crosses back through 20-period EMA
    - Uses daily trend to filter reversals in ranging markets
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h Keltner channels (20, 1.5)
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_series = pd.Series(np.maximum(high - low, np.maximum(abs(high - close_series.shift(1)), abs(low - close_series.shift(1)))))
    atr = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    upper_keltner = ema20 + 1.5 * atr
    lower_keltner = ema20 - 1.5 * atr
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Close below lower Keltner + daily uptrend + volume spike
            if close[i] < lower_keltner[i] and close[i] > ema50_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Close above upper Keltner + daily downtrend + volume spike
            elif close[i] > upper_keltner[i] and close[i] < ema50_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close crosses back through 20-period EMA
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses back through 20-period EMA
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals