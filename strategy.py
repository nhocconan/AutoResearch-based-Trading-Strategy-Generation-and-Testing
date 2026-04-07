# 2027: 6h_Donchian20_WeeklyPivot_Direction_Volume
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Weekly pivot levels from previous week: buy at S1/S2 in uptrend, sell at R1/R2 in downtrend.
# Uses 6h timeframe to balance trade frequency and signal quality. Target: 50-150 trades over 4 years.
# Works in bull/bear by aligning breakout direction with higher timeframe trend via weekly pivot position.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_direction_volume"
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
    
    # === WEEKLY PIVOT (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Weekly pivot calculation (using previous week's data)
    w_pivot = (w_high + w_low + w_close * 2) / 4
    w_range = w_high - w_low
    
    # Weekly support/resistance levels
    R1 = 2 * w_pivot - w_low
    S1 = 2 * w_pivot - w_high
    R2 = w_pivot + w_range
    S2 = w_pivot - w_range
    
    # Align weekly levels to 6h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # === 6H DONCHIAN CHANNEL (LTF) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime from weekly pivot position
        # Price above weekly pivot = bullish bias, below = bearish bias
        bullish_bias = close[i] > w_pivot[-1] if len(w_pivot) > 0 else False  # Use last known weekly pivot
        
        if position == 1:  # Long position
            # Exit: price retests weekly S1 or breaks below 6h low
            if close[i] <= S1_aligned[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests weekly R1 or breaks above 6h high
            if close[i] >= R1_aligned[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic: Donchian breakout with weekly pivot filter
            if close[i] > highest_high[i]:  # Bullish breakout
                # Only take long if bullish bias or price above weekly pivot
                if bullish_bias or close[i] > w_pivot[-1]:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < lowest_low[i]:  # Bearish breakout
                # Only take short if bearish bias or price below weekly pivot
                if not bullish_bias or close[i] < w_pivot[-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals