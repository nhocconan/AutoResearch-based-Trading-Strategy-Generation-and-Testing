#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + volume confirmation + 1d EMA34 trend filter
# Long when price breaks above R3 with volume > 1.5x 20-period avg and 1d EMA34 upward
# Short when price breaks below S3 with volume > 1.5x 20-period avg and 1d EMA34 downward
# Exit when price crosses opposite Camarilla level or trend reverses
# Targets 20-50 trades per year for optimal fee drag (< 200 total over 4 years)
# Camarilla provides clear support/resistance, volume confirms momentum, EMA34 filters counter-trend noise

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # Camarilla levels (from previous day)
    def calculate_camarilla(prev_high, prev_low, prev_close):
        range_val = prev_high - prev_low
        r3 = prev_close + range_val * 1.1 / 2
        r2 = prev_close + range_val * 1.1 / 4
        r1 = prev_close + range_val * 1.1 / 6
        s1 = prev_close - range_val * 1.1 / 6
        s2 = prev_close - range_val * 1.1 / 4
        s3 = prev_close - range_val * 1.1 / 2
        return r1, r2, r3, s1, s2, s3
    
    # Calculate Camarilla levels for each day
    r1_series = np.full(n, np.nan)
    r2_series = np.full(n, np.nan)
    r3_series = np.full(n, np.nan)
    s1_series = np.full(n, np.nan)
    s2_series = np.full(n, np.nan)
    s3_series = np.full(n, np.nan)
    
    # Group by date to get previous day's OHLC
    dates = pd.to_datetime(prices['open_time']).dt.date
    unique_dates = np.unique(dates)
    
    for date in unique_dates:
        mask = dates == date
        if not np.any(mask):
            continue
        
        # Get previous day's data
        prev_date = date - pd.Timedelta(days=1)
        prev_mask = dates == prev_date
        if not np.any(prev_mask):
            # First day, use current day's data as placeholder
            prev_high = high[mask].max()
            prev_low = low[mask].min()
            prev_close = close[mask][-1]
        else:
            prev_high = high[prev_mask].max()
            prev_low = low[prev_mask].min()
            prev_close = close[prev_mask][-1]
        
        r1, r2, r3, s1, s2, s3 = calculate_camarilla(prev_high, prev_low, prev_close)
        
        # Set levels for current day
        r1_series[mask] = r1
        r2_series[mask] = r2
        r3_series[mask] = r3
        s1_series[mask] = s1
        s2_series[mask] = s2
        s3_series[mask] = s3
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
    
    start_idx = 50  # Need enough data for Camarilla and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_series[i]) or np.isnan(s3_series[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        r3_val = r3_series[i]
        s3_val = s3_series[i]
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