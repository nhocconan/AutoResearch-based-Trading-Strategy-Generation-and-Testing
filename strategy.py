# 6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot breakouts at R4/S4 levels with daily trend filter and volume confirmation.
# In 2025 bear market, institutional breakouts at extreme levels (R4/S4) with volume and trend confirmation
# capture momentum while avoiding false breakouts. Works in bull (continuation) and bear (sharp reversals at extremes).
# Target: 15-30 trades/year to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels: R4 = close + 1.5 * (high - low), S4 = close - 1.5 * (high - low)
    camarilla_r4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_s4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema_1d = ema_20_1d_aligned[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > R4 AND price > daily EMA20 (uptrend) AND volume > 2x average
            if close[i] > r4 and close[i] > ema_1d and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S4 AND price < daily EMA20 (downtrend) AND volume > 2x average
            elif close[i] < s4 and close[i] < ema_1d and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S4 OR trend reverses (price < daily EMA20)
            if close[i] < s4 or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R4 OR trend reverses (price > daily EMA20)
            if close[i] > r4 or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals