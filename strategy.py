#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + Daily Trend + Volume Spike
# Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
# Fade at R3/S3 levels during ranging markets, breakout continuation at R4/S4 during trending markets.
# Uses 60-period EMA on daily for trend filter and volume confirmation to avoid false signals.
# Designed for 6h timeframe to target 12-35 trades/year (50-140 total over 4 years).

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # where H,L,C are previous day's high, low, close
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily EMA(60) for trend filter
    ema_60_1d = pd.Series(close_1d).ewm(span=60, adjust=False).mean().values
    ema_60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_60_1d)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_60_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or trend turns bearish
            if close[i] < s3_aligned[i] or close[i] < ema_60_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or trend turns bullish
            if close[i] > r3_aligned[i] or close[i] > ema_60_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Fade at R3/S3 in ranging market (price near extremes but trend weak)
                # OR breakout at R4/S4 in trending market
                if ((close[i] <= r3_aligned[i] and close[i] >= s3_aligned[i]) and  # In R3-S3 range
                    (abs(close[i] - ema_60_1d_aligned[i]) < (r3_aligned[i] - s3_aligned[i]) * 0.3)):  # Near middle
                    # Fade logic: buy near S3, sell near R3
                    if close[i] <= s3_aligned[i] * 1.002:  # Near S3 (0.2% buffer)
                        position = 1
                        signals[i] = 0.25
                    elif close[i] >= r3_aligned[i] * 0.998:  # Near R3 (0.2% buffer)
                        position = -1
                        signals[i] = -0.25
                elif close[i] > r4_aligned[i] and close[i] > ema_60_1d_aligned[i]:  # Breakout above R4 in uptrend
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_aligned[i] and close[i] < ema_60_1d_aligned[i]:  # Breakdown below S4 in downtrend
                    position = -1
                    signals[i] = -0.25
    
    return signals