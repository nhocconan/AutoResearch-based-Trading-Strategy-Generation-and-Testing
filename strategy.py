#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + 1d Trend + Volume Spike
# Hypothesis: Camarilla levels from 1d provide precise S/R zones. 
# In ranging markets, fade at R3/S3 (80% retracement levels). 
# In trending markets, breakout continuation at R4/S4 (breakout levels).
# 1d EMA50 filters trend direction. Volume spikes confirm institutional participation.
# Works in bull: buy at S3/S4 in uptrend, sell at R3/R4.
# Works in bear: sell at R3/R4 in downtrend, buy at S3/S4.
# Target: 12-37 trades/year (50-150 total over 4 years) for 6h timeframe.

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels for each day (using previous day's OHLC)
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    cam_r4 = close_1d + 1.5 * (high_1d - low_1d)
    cam_r3 = close_1d + 1.1 * (high_1d - low_1d)
    cam_s3 = close_1d - 1.1 * (high_1d - low_1d)
    cam_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day for non-look-ahead)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    # Volume confirmation: volume > 1.8x 30-period average (balanced threshold)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cam_r4_aligned[i]) or 
            np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(cam_s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or trend turns bearish
            if close[i] >= cam_r3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or trend turns bullish
            if close[i] <= cam_s3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Fade at S3/S3 in uptrend (mean reversion)
                if close[i] <= cam_s3_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Fade at R3/R3 in downtrend (mean reversion)
                elif close[i] >= cam_r3_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                # Breakout continuation at S4 in uptrend
                elif close[i] <= cam_s4_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout continuation at R4 in downtrend
                elif close[i] >= cam_r4_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals