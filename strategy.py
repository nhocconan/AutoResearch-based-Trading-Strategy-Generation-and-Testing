#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with volume spike and 1d EMA34 trend filter
# Long when price breaks above R3 with volume > 1.5x avg and 1d EMA34 rising
# Short when price breaks below S3 with volume > 1.5x avg and 1d EMA34 falling
# Exit when price crosses opposite level (S3 for long, R3 for short) or trend reverses
# Targets 12-37 trades/year for minimal fee drag (< 200 total over 4 years)
# Camarilla provides institutional levels, volume confirms breakout strength, EMA34 filters noise

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels using prior day's OHLC
    # Camarilla formulas: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C, H, L are from previous 12h period (daily for 12h timeframe)
    # We'll use prior 12h bar's high/low/close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    hl_range = prev_high - prev_low
    r3 = prev_close + hl_range * 1.1 / 2
    s3 = prev_close - hl_range * 1.1 / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_slope = ema34_1d[1:] - ema34_1d[:-1]  # slope: positive = uptrend
    ema34_1d_slope = np.concatenate([[0], ema34_1d_slope])  # align length
    
    # Align 1d EMA34 and slope to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for EMA34 and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        r3_val = r3[i]
        s3_val = s3[i]
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