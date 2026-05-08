#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with volume confirmation and 1d EMA34 trend filter
# Long when price breaks above R3 with volume > 1.5x 20-period avg and 1d EMA34 upward
# Short when price breaks below S3 with volume > 1.5x 20-period avg and 1d EMA34 downward
# Exit when price crosses opposite level (S3 for long, R3 for short) or trend reverses
# Camarilla levels provide institutional support/resistance, volume confirms momentum, EMA34 filters counter-trend noise
# Targets 25-50 trades per year for optimal drag (< 200 total over 4 years)

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # R3 = H + 2*(H-L)/1.1, S3 = L - 2*(H-L)/1.1
    # We need to calculate these once per day and carry forward
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Group by date to calculate daily levels
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    for date in unique_dates:
        mask = (dates == date)
        if not np.any(mask):
            continue
        # Get previous day's data
        day_idx = np.where(dates == date)[0]
        if day_idx[0] == 0:
            continue  # No previous day
        prev_day_idx = day_idx[0] - 1
        prev_high = high[prev_day_idx]
        prev_low = low[prev_day_idx]
        prev_close = close[prev_day_idx]
        
        # Calculate Camarilla levels for current day
        r3 = prev_high + 2 * (prev_high - prev_low) / 1.1
        s3 = prev_low - 2 * (prev_high - prev_low) / 1.1
        
        camarilla_r3[mask] = r3
        camarilla_s3[mask] = s3
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_slope = ema34_1d[1:] - ema34_1d[:-1]  # slope: positive = uptrend
    ema34_1d_slope = np.concatenate([[0], ema34_1d_slope])  # align length
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        r3_val = camarilla_r3[i]
        s3_val = camarilla_s3[i]
        ema34_val = ema34_1d_aligned[i]
        ema34_slope = ema34_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: break above R3, volume confirmation, 1d uptrend (positive slope)
            if close_val > r3_val and vol_conf_val and ema34_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3, volume confirmation, 1d downtrend (negative slope)
            elif close_val < s3_val and vol_conf_val and ema34_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S3 or 1d trend turns down
            if close_val < s3_val or ema34_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R3 or 1d trend turns up
            if close_val > r3_val or ema34_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals