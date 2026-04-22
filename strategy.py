# 12h_Camarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeSpike_v1
# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike (2x 20-period MA)
# Camarilla R4/S4 levels are stronger institutional support/resistance than R3/S3, reducing false breakouts
# 1d EMA50 filters for longer-term trend direction, avoiding counter-trend trades
# Volume spike confirms institutional participation
# Works in bull/bear: breaks through key levels with trend and volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Camarilla multiplier for R4/S4 levels
    camarilla_multiplier = 1.1 / 6
    
    # Calculate Camarilla R4 and S4 levels for each 1d bar (based on previous day)
    camarilla_r4 = close_1d + (high_1d - low_1d) * camarilla_multiplier * 4
    camarilla_s4 = close_1d - (high_1d - low_1d) * camarilla_multiplier * 4
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R4 with volume spike and price above 1d EMA50 (uptrend)
            if close[i] > camarilla_r4_aligned[i] and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S4 with volume spike and price below 1d EMA50 (downtrend)
            elif close[i] < camarilla_s4_aligned[i] and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Camarilla level (S4 for longs, R4 for shorts)
            if position == 1:
                if close[i] < camarilla_s4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_r4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0