#52
#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Uses 4h timeframe with 12h trend (EMA50) and volume confirmation to filter Camarilla R1/S1 breakouts.
# Enters long when price breaks above daily R1 in 12h uptrend (close > EMA50) with volume > 1.5x 20-period average.
# Enters short when price breaks below daily S1 in 12h downtrend (close < EMA50) with volume confirmation.
# Exits when price returns to opposite level (S1 for long, R1 for short) or trend reverses.
# Uses 12h EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 25-50 trades per year on 4h timeframe with position size 0.25.

name = "4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot_range = prev_high - prev_low
    r1_level = prev_close + 1.1 * (pivot_range / 12)  # R1 = C + 1.1*(H-L)/12
    s1_level = prev_close - 1.1 * (pivot_range / 12)  # S1 = C - 1.1*(H-L)/12
    
    # Align pivot levels to 4h timeframe (available after 1d bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume filter: volume > 1.5x 20-period average on 4h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 in 12h uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 in 12h downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to S1 or trend reverses to downtrend
            if (close[i] < s1_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to R1 or trend reverses to uptrend
            if (close[i] > r1_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals