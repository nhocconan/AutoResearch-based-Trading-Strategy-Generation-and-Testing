#!/usr/bin/env python3
"""
12h Monthly Pivot S1/S2 Breakout with Volume Spike and 1d EMA34 Trend Filter
Hypothesis: Monthly pivot levels (S2/S1) act as strong support/resistance across market regimes.
Price breaking below S2 with volume surge and below 1d EMA34 indicates bearish momentum.
Price breaking above S1 with volume surge and above 1d EMA34 indicates bullish momentum.
Trend filter ensures alignment with higher timeframe direction. Designed for low frequency (<30 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get monthly data for pivot levels (once before loop)
    df_m = get_htf_data(prices, '1M')
    
    # Calculate monthly high, low, close for pivot levels
    monthly_high = df_m['high'].values
    monthly_low = df_m['low'].values
    monthly_close = df_m['close'].values
    
    # Calculate monthly pivot: P = (H+L+C)/3
    monthly_pivot = (monthly_high + monthly_low + monthly_close) / 3.0
    monthly_range = monthly_high - monthly_low
    # Monthly S2 = P - 2*(H-L)
    monthly_s2 = monthly_pivot - 2.0 * monthly_range
    # Monthly S1 = P - (H-L)
    monthly_s1 = monthly_pivot - monthly_range
    # Monthly R1 = P + (H-L)
    monthly_r1 = monthly_pivot + monthly_range
    # Monthly R2 = P + 2*(H-L)
    monthly_r2 = monthly_pivot + 2.0 * monthly_range
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align monthly pivot levels to 12h timeframe
    s2_aligned = align_htf_to_ltf(prices, df_m, monthly_s2)
    s1_aligned = align_htf_to_ltf(prices, df_m, monthly_s1)
    r1_aligned = align_htf_to_ltf(prices, df_m, monthly_r1)
    r2_aligned = align_htf_to_ltf(prices, df_m, monthly_r2)
    
    # Volume spike detection (2.0x 24-period average to reduce frequency)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(s2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        s2_level = s2_aligned[i]
        s1_level = s1_aligned[i]
        r1_level = r1_aligned[i]
        r2_level = r2_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        
        if position == 0:
            # Bearish: break below monthly S2 with volume spike and below daily EMA34
            if (price < s2_level and volume_spike[i] and price < ema_34):
                signals[i] = -0.25
                position = -1
            # Bullish: break above monthly S1 with volume spike and above daily EMA34
            elif (price > s1_level and volume_spike[i] and price > ema_34):
                signals[i] = 0.25
                position = 1
        
        elif position == -1:
            # Short position: hold until reversal signal
            signals[i] = -0.25
            # Reverse to long: price breaks above monthly R1 with volume and above EMA34
            if (price > r1_level and volume_spike[i] and price > ema_34):
                signals[i] = 0.25
                position = 1
            # Exit to flat: price returns to monthly S1 (support hold)
            elif price >= s1_level:
                signals[i] = 0.0
                position = 0
        
        elif position == 1:
            # Long position: hold until reversal signal
            signals[i] = 0.25
            # Reverse to short: price breaks below monthly S2 with volume and below EMA34
            if (price < s2_level and volume_spike[i] and price < ema_34):
                signals[i] = -0.25
                position = -1
            # Exit to flat: price returns to monthly R1 (resistance hold)
            elif price <= r1_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_MonthlyPivot_S1S2_Breakout_VolumeSpike_1dEMA34"
timeframe = "12h"
leverage = 1.0