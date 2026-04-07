#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot Reversal with Volume Confirmation and 12h Trend Filter
# Hypothesis: Price reversals at Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts)
# offer high-probability entries when confirmed by volume spikes and aligned with 12h trend.
# Works in bull markets (buy dips at S3/S4 in uptrend) and bear markets (sell rallies at R3/R4 in downtrend).
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag.

name = "4h_camarilla_pivot_reversal_volume_trend_v1"
timeframe = "4h"
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
    
    # Get daily data for Camarilla calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_daily['close'].values
    prev_high = df_daily['high'].values
    prev_low = df_daily['low'].values
    prev_range = prev_high - prev_low
    
    # Camarilla levels (based on previous day)
    # S1 = C - (H-L)*1.08/2, S2 = C - (H-L)*1.16/2, S3 = C - (H-L)*1.24/2, S4 = C - (H-L)*1.50/2
    # R1 = C + (H-L)*1.08/2, R2 = C + (H-L)*1.16/2, R3 = C + (H-L)*1.24/2, R4 = C + (H-L)*1.50/2
    s3 = prev_close - (prev_range * 1.24 / 2)
    s4 = prev_close - (prev_range * 1.50 / 2)
    r3 = prev_close + (prev_range * 1.24 / 2)
    r4 = prev_close + (prev_range * 1.50 / 2)
    
    # Handle first day
    if len(s3) > 1:
        s3[0] = s3[1]
        s4[0] = s4[1]
        r3[0] = r3[1]
        r4[0] = r4[1]
    else:
        s3[0] = s3[0] if len(s3) > 0 else 0
        s4[0] = s4[0] if len(s4) > 0 else 0
        r3[0] = r3[0] if len(r3) > 0 else 0
        r4[0] = r4[0] if len(r4) > 0 else 0
    
    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    
    # 12h trend filter: EMA50 for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Volume filter: volume > 2.0x 20-period average (strict to reduce trades)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below S4 or trend turns bearish or volume drops
            if (low[i] < s4_aligned[i] or close[i] < ema_12h_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above R4 or trend turns bullish or volume drops
            if (high[i] > r4_aligned[i] or close[i] > ema_12h_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price closes below S3 (deep oversold) with volume and bullish 12h trend
            if (close[i] < s3_aligned[i] and close[i] > ema_12h_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price closes above R3 (deep overbought) with volume and bearish 12h trend
            elif (close[i] > r3_aligned[i] and close[i] < ema_12h_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals