#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels (R3/S3) from 1h chart for precise entry/exit, 4h EMA50 for trend direction,
# and 1h volume spike (2x 20-period average) to confirm momentum. Designed for 15-35 trades/year
# (~60-140 total over 4 years) to minimize fee drag. Session filter (08-20 UTC) reduces noise.
# Works in bull/bear markets by aligning with 4h trend and requiring volume confirmation.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA50
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 4h EMA50 to 1h timeframe (wait for completed 4h bar)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h Camarilla pivots (R3, S3) - using previous bar's H/L/C
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We need to shift by 1 to use only completed bar data for pivot calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Calculate 1h volume spike (2x 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, above 4h EMA50, and volume spike
            if (close[i] > r3[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3, below 4h EMA50, and volume spike
            elif (close[i] < s3[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla H-L range OR breaks below S3
            if (close[i] <= prev_high[i] and close[i] >= prev_low[i]) or close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price re-enters Camarilla H-L range OR breaks above R3
            if (close[i] <= prev_high[i] and close[i] >= prev_low[i]) or close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals