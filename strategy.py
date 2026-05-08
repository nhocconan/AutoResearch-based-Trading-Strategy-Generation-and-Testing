#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w trend filter and volume confirmation
# Long when price closes above R3, 1w EMA34 rising, volume > 1.5x average
# Short when price closes below S3, 1w EMA34 falling, volume > 1.5x average
# Uses Camarilla pivot levels for institutional support/resistance, EMA34 for trend filter
# Targets 8-25 trades per year (32-100 over 4 years) to avoid fee drag
# Works in both bull and bear markets due to trend filter and volume confirmation

name = "1d_Camarilla_R3S3_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate previous day's typical price for Camarilla (avoid look-ahead)
    typical_price = (high + low + close) / 3
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan
    
    # Calculate Camarilla levels from previous day
    # R3 = close + 1.1 * (high - low)  [using previous day's values]
    # S3 = close - 1.1 * (high - low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Calculate EMA34 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 days of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        r3_val = camarilla_r3[i]
        s3_val = camarilla_s3[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: close above R3, 1w uptrend, volume confirmation
            if close_val > r3_val and ema34_1w_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: close below S3, 1w downtrend, volume confirmation
            elif close_val < s3_val and ema34_1w_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below S3 or 1w trend down
            if close_val < s3_val or ema34_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above R3 or 1w trend up
            if close_val > r3_val or ema34_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals