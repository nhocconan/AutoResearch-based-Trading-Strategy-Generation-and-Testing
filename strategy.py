#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3, 1d EMA34 up-trend, volume > 2.5x 20-period average
# Short when price breaks below Camarilla S3, 1d EMA34 down-trend, volume > 2.5x 20-period average
# Exit when price crosses the 50% level (Camarilla midpoint)
# Uses discrete position sizing (0.25) and strict volume filter to target 20-40 trades/year.
# Camarilla levels from daily timeframe provide strong intraday support/resistance that works in both bull and bear markets.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    daily_range = high_1d - low_1d
    camarilla_h5 = close_1d + daily_range * 1.1/2  # R3 equivalent
    camarilla_h4 = close_1d + daily_range * 1.1/4  # R2 equivalent
    camarilla_h3 = close_1d + daily_range * 1.1/6  # R1 equivalent
    camarilla_l3 = close_1d - daily_range * 1.1/6  # S1 equivalent
    camarilla_l2 = close_1d - daily_range * 1.1/4  # S2 equivalent
    camarilla_l1 = close_1d - daily_range * 1.1/2  # S3 equivalent
    camarilla_mid = close_1d  # Pivot point (close of previous day)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    camarilla_h5 = np.roll(camarilla_h5, 1)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_l2 = np.roll(camarilla_l2, 1)
    camarilla_l1 = np.roll(camarilla_l1, 1)
    camarilla_mid = np.roll(camarilla_mid, 1)
    camarilla_h5[0] = np.nan
    camarilla_h4[0] = np.nan
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    camarilla_l2[0] = np.nan
    camarilla_l1[0] = np.nan
    camarilla_mid[0] = np.nan
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed for pivot points)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # EMA34 and volume warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_h5 = camarilla_h5_aligned[i]   # R3
        curr_l1 = camarilla_l1_aligned[i]   # S3
        curr_mid = camarilla_mid_aligned[i]  # Pivot
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below midpoint (mean reversion to pivot)
            if curr_close < curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above midpoint (mean reversion to pivot)
            if curr_close > curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.5x 20-period average (strict filter)
            vol_confirmed = curr_volume > 2.5 * curr_vol_ma
            
            # Long when price breaks above H5 (R3), 1d EMA34 up-trend, volume confirmed
            if curr_high > curr_h5 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below L1 (S3), 1d EMA34 down-trend, volume confirmed
            elif curr_low < curr_l1 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals