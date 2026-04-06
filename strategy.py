#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot with 1d trend filter and volume confirmation.
# Long when price crosses above R3 (bullish bias) and 1d close > EMA50 (bullish trend).
# Short when price crosses below S3 (bearish bias) and 1d close < EMA50 (bearish trend).
# Uses volume > 1.3x 20-period average for confirmation.
# Camarilla levels from prior day provide intraday support/resistance.
# Trend filter prevents counter-trend trades. Target: 75-150 total trades over 4 years.

name = "6h_camarilla_pivot_1d_trend_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Need daily high, low, close from prior day
    df_1d = get_htf_data(prices, '1d')
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d trend filter: EMA50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if Camarilla or EMA data not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below S3 or trend turns bearish
            if (low[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above R3 or trend turns bullish
            if (high[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: price crosses above R3 during bullish trend
                if (high[i] > camarilla_r3_aligned[i] and 
                    close[i] > ema_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price crosses below S3 during bearish trend
                elif (low[i] < camarilla_s3_aligned[i] and 
                      close[i] < ema_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals