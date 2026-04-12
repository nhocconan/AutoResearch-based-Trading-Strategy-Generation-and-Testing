# 12h_1d_camarilla_pivot_volume_v1
# Hypothesis: Camarilla pivot levels on daily timeframe act as strong support/resistance levels.
# Price rejection at these levels with volume confirmation provides high-probability mean-reversion
# entries. Works in both bull and bear markets as pivot levels adapt to volatility.
# Target: 20-50 trades/year on 12h timeframe with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    camarilla_r4 = close_1d + range_hl * 1.1 / 2.0
    camarilla_r3 = close_1d + range_hl * 1.1 / 4.0
    camarilla_r2 = close_1d + range_hl * 1.1 / 6.0
    camarilla_r1 = close_1d + range_hl * 1.1 / 12.0
    camarilla_s1 = close_1d - range_hl * 1.1 / 12.0
    camarilla_s2 = close_1d - range_hl * 1.1 / 6.0
    camarilla_s3 = close_1d - range_hl * 1.1 / 4.0
    camarilla_s4 = close_1d - range_hl * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: current 12h volume > 1.5 * 20-period MA
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_r2_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion at extreme Camarilla levels
        # Long when price touches or goes below S3/S4 with volume
        long_touch = (close[i] <= camarilla_s3_aligned[i] or close[i] <= camarilla_s4_aligned[i]) and vol_filter[i]
        # Short when price touches or goes above R3/R4 with volume
        short_touch = (close[i] >= camarilla_r3_aligned[i] or close[i] >= camarilla_r4_aligned[i]) and vol_filter[i]
        
        # Exit when price returns to mean (typical price) or opposite extreme
        typical_price_today = (high[i] + low[i] + close[i]) / 3.0
        typical_price_yesterday = typical_price[i-1] if i > 0 else typical_price[0]
        
        long_exit = close[i] >= typical_price_yesterday
        short_exit = close[i] <= typical_price_yesterday
        
        if long_touch and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_touch and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0