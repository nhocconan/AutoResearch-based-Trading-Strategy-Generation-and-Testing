# 1d_Camarilla_R1_S1_Breakout_1wTrend
# Hypothesis: Use 1d Camarilla R1/S1 breakouts with 1w EMA trend filter for multi-day persistence.
# In bull markets: buy breakouts above R1 when above weekly EMA; in bear markets: sell breakdowns below S1 when below weekly EMA.
# Weekly EMA provides trend filter that reduces whipsaws in ranging markets while capturing strong trends.
# Target: 20-40 trades/year to stay under fee drag threshold.
# Uses 1w EMA for trend filter (not 1d) to avoid overtrading on lower timeframe.
# Volume confirmation ensures breakouts have conviction.
# Stops when price crosses back through pivot point (PP) or volume dries up.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data once for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels (based on previous day's HLC)
    # Note: using previous day's levels to avoid look-ahead
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # 1w EMA34 for trend filter (using weekly close)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 1d timeframe (since we're using 1d timeframe)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pp = pp_aligned[i]
        ema34 = ema34_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above R1 + volume spike + price > weekly EMA
            if price > r1 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 + volume spike + price < weekly EMA
            elif price < s1 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through PP or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below PP or volume dries up
                if price < pp or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above PP or volume dries up
                if price > pp or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0