#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike.
# Long when price breaks above R3 in bull trend (close > 12h EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below S3 in bear trend (close < 12h EMA50) with volume spike.
# Camarilla levels provide structure; 12h EMA50 filters whipsaw; volume confirms participation.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # Using previous 12h bar's high, low, close
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    camarilla_r3 = c_12h + (h_12h - l_12h) * 1.1 / 4
    camarilla_s3 = c_12h - (h_12h - l_12h) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if is_bull_trend and close_val > r3 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and close_val < s3 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR trend reversal
            if close_val < s3 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR trend reversal
            if close_val > r3 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals