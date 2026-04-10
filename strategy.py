#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w Williams %R regime filter
# - Primary: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures buying/selling pressure
# - Entry: Long when Bull Power > 0 and rising, Bear Power < 0 and falling (strong bullish momentum)
#          Short when Bear Power < 0 and falling, Bull Power > 0 and rising (strong bearish momentum)
# - Regime filter: 1w Williams %R < -80 for long (oversold weekly), > -20 for short (overbought weekly)
# - Exit: Elder Ray divergence - when Bull Power peaks and turns down for longs, Bear Power troughs and turns up for shorts
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Williams %R adapts to weekly extremes, Elder Ray measures intrinsic strength

name = "6h_1w_elder_ray_williams_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Buying pressure
    bear_power = low - ema13   # Selling pressure
    
    # Calculate 1w Williams %R for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          -(highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14) * 100,
                          -50)  # neutral when range is zero
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and rising, Williams %R oversold (< -80)
            if (bull_power[i] > 0 and 
                i > 0 and bull_power[i] > bull_power[i-1] and
                williams_r_aligned[i] < -80):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 and falling, Williams %R overbought (> -20)
            elif (bear_power[i] < 0 and 
                  i > 0 and bear_power[i] < bear_power[i-1] and
                  williams_r_aligned[i] > -20):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit on Elder Ray divergence or extreme reversal
            if position == 1:  # Long position
                # Exit when Bull Power peaks and turns down OR Williams %R becomes overbought
                if (i > 0 and bull_power[i] < bull_power[i-1]) or williams_r_aligned[i] > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Exit when Bear Power troughs and turns up OR Williams %R becomes oversold
                if (i > 0 and bear_power[i] > bear_power[i-1]) or williams_r_aligned[i] < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals