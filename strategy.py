#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 in bull trend (close > 1d EMA50) with volume spike.
# Short when price breaks below Camarilla S3 in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Camarilla pivot levels provide high-probability intraday support/resistance.
# The 1d EMA50 filter ensures alignment with higher timeframe trend.
# Volume confirmation reduces false signals. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1dEMA50_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need at least 50 for EMA + 1 for current
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d_prev = df_1d.copy()
    # Calculate Camarilla levels from previous day's data
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But standard Camarilla uses: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # where high, low, close are from previous day
    high_1d = df_1d_prev['high'].values
    low_1d = df_1d_prev['low'].values
    close_1d = df_1d_prev['close'].values
    
    # Calculate Camarilla R3 and S3 for each 1d bar
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_s3)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if is_bull_trend and close_val > r3_level and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and close_val < s3_level and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S3 OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla R3 OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals