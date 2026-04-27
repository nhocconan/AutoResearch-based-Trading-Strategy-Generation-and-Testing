#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike
Hypothesis: Uses 4h Camarilla pivot levels (R1/S1) for breakout entries in the direction of 4h trend (price > EMA34). Volume confirmation (>1.5x 20-period average on 1d) ensures conviction. 1h timeframe for precise entry timing, with 4h for signal direction and 1d for volume regime filter. Targets 15-35 trades/year by requiring confluence of HTF trend, HTF breakout, and HTF volume spike. Works in bull markets via upside breakouts and in bear markets via downside breakdowns. Camarilla R1/S1 represent tight intraday levels where breakouts often continue, while the 4h EMA34 filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h Camarilla pivot levels (using previous 4h bar's HLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Shift by 1 to use previous 4h bar's HLC for current levels
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    # Camarilla levels (R1/S1, R3/S3 for exits)
    range_4h = prev_high_4h - prev_low_4h
    camarilla_r1 = prev_close_4h + range_4h * 1.1 / 12
    camarilla_s1 = prev_close_4h - range_4h * 1.1 / 12
    camarilla_r3 = prev_close_4h + range_4h * 1.1 / 4
    camarilla_s3 = prev_close_4h - range_4h * 1.1 / 4
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_avg_1d)
    
    # Align all HTF indicators to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital (discrete level to reduce churn)
    
    # Warmup: need EMA34 (34), volume avg (20), and Camarilla (need previous bar)
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_4h_val = ema_34_4h_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        if position == 0:
            # Determine 4h trend: price > EMA34 = uptrend, price < EMA34 = downtrend
            is_uptrend = close_val > ema_4h_val
            is_downtrend = close_val < ema_4h_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R1 and volume spike confirms
                if (close_val > r1) and vol_spike:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S1 and volume spike confirms
                if (close_val < s1) and vol_spike:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price reverts to R3 or trend changes to downtrend
            exit_condition = (close_val < r3) or (close_val < ema_4h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reverts to S3 or trend changes to uptrend
            exit_condition = (close_val > s3) or (close_val > ema_4h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0