#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 in bull trend (close > 1w EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below S3 in bear trend (close < 1w EMA50) with volume spike.
# Daily timeframe reduces trade frequency to avoid fee drag; 1w EMA50 filters whipsaw.
# Camarilla levels provide structure; volume confirms participation.
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25.
# Works in both bull and bear by following the 1w trend direction.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla levels (based on previous 1d bar)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # Using previous 1d bar's high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
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