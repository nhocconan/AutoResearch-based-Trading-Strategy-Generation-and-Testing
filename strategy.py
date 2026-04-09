#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot levels + volume confirmation
# - Long when price breaks above weekly R4 with volume > 2.0x 20-period average
# - Short when price breaks below weekly S4 with volume > 2.0x 20-period average
# - Exit when price returns to weekly pivot point (PP) or opposite Camarilla level (S4 for long, R4 for short)
# - Weekly Camarilla levels provide structural support/resistance that work in both bull and bear markets
# - Volume confirmation filters false breakouts
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Camarilla formulas:
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    # R2 = PP + (H - L) * 1.1/6
    # S2 = PP - (H - L) * 1.1/6
    # R1 = PP + (H - L) * 1.1/12
    # S1 = PP - (H - L) * 1.1/12
    
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    pp_1w = typical_price_1w
    r4_1w = pp_1w + range_1w * 1.1 / 2.0
    s4_1w = pp_1w - range_1w * 1.1 / 2.0
    r3_1w = pp_1w + range_1w * 1.1 / 4.0
    s3_1w = pp_1w - range_1w * 1.1 / 4.0
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed weekly bar)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions: price returns to pivot point or breaks below S3
            if close[i] <= pp_aligned[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price returns to pivot point or breaks above R3
            if close[i] >= pp_aligned[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: weekly Camarilla breakout + volume confirmation
            if volume_confirmed:
                # Long entry: price breaks above weekly R4
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below weekly S4
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals