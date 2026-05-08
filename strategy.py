#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 reversal with 12h EMA34 trend filter and volume confirmation
# Long when price breaks below S3 in a 12h uptrend with volume spike (mean reversion in uptrend)
# Short when price breaks above R3 in a 12h downtrend with volume spike (mean reversion in downtrend)
# Uses Camarilla levels for mean reversion zones, EMA34 for trend filter, volume for confirmation
# Designed for ranging markets with clear trend context to avoid counter-trend traps
# Targets 12-37 trades per year (50-150 over 4 years) for low fee drift

name = "6h_Camarilla_R3S3_Reversion_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's OHLC (to avoid look-ahead)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_open = np.roll(df_12h['open'].values, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_open[0] = np.nan
    
    # Calculate Camarilla levels for previous 12h bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    rng = prev_high - prev_low
    r3 = prev_close + 1.1 * rng
    s3 = prev_close - 1.1 * rng
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    
    # Calculate EMA34 on 12h close for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need at least 34 bars for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        ema34_val = ema34_12h_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks below S3 in 12h uptrend with volume spike
            if close_val < s3_val and ema34_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks above R3 in 12h downtrend with volume spike
            elif close_val > r3_val and ema34_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above S3 (mean reversion complete) or trend turns down
            if close_val > s3_val or ema34_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below R3 (mean reversion complete) or trend turns up
            if close_val < r3_val or ema34_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals