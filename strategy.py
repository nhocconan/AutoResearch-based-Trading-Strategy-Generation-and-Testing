#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability support/resistance derived from prior day's range.
# Breakouts above R3 in uptrend (price > EMA34) or below S3 in downtrend capture strong moves
# with controlled trade frequency. Volume spike confirms conviction. Designed for 12-25 trades/year
# on 12h to minimize fee drag while maintaining edge in bull/bear markets via trend alignment.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of completed daily bar)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_R3 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 2
    camarilla_S3 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (available after daily bar closes)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from bar 1 to have prior daily data
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2x 20-bar EMA of volume
        vol_lookback = min(20, i+1)
        vol_slice = volume[max(0, i-vol_lookback+1):i+1]
        if len(vol_slice) > 0:
            vol_ema = pd.Series(vol_slice).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
            volume_spike = volume[i] > (2.0 * vol_ema)
        else:
            volume_spike = False
        
        breakout_long = close[i] > camarilla_R3_aligned[i]
        breakout_short = close[i] < camarilla_S3_aligned[i]
        
        if position == 0:
            # Long: break above R3 in 1d uptrend with volume spike
            if breakout_long and ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in 1d downtrend with volume spike
            elif breakout_short and ema_34_1d_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 or loses 1d uptrend
            if close[i] < camarilla_R3_aligned[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above S3 or loses 1d downtrend
            if close[i] > camarilla_S3_aligned[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals