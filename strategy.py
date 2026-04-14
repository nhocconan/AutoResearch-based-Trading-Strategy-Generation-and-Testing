#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with weekly trend filter
# Weekly EMA(50) determines trend direction to avoid counter-trend trades
# At 6h, price reverses from Camarilla R3/S3 levels in direction of weekly trend
# Volume > 1.3x average confirms institutional participation at reversal points
# Works in bull/bear as weekly EMA adapts to long-term trend
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend filter
    ema_len = 50
    if len(df_1w) < ema_len:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load daily data ONCE for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla pivot calculation (based on previous day)
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use previous day's H,L,C to calculate today's levels
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for current day
    camarilla_h = prev_close + ((prev_high - prev_low) * 1.1 / 2)  # R4
    camarilla_l = prev_close - ((prev_high - prev_low) * 1.1 / 2)  # S4
    camarilla_3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)  # R3/S3
    camarilla_4 = prev_close - ((prev_high - prev_low) * 1.1 / 4)  # S3/R3
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h)
    camarilla_l_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l)
    camarilla_3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_3)
    camarilla_4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_4)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or
            np.isnan(camarilla_h_aligned[i]) or
            np.isnan(camarilla_l_aligned[i]) or
            np.isnan(camarilla_3_aligned[i]) or
            np.isnan(camarilla_4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA50
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price at S3/S4 + above weekly EMA + volume
            if ((close[i] <= camarilla_4_aligned[i] or close[i] <= camarilla_3_aligned[i]) and
                above_weekly_ema and
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price at R3/R4 + below weekly EMA + volume
            elif ((close[i] >= camarilla_3_aligned[i] or close[i] >= camarilla_h_aligned[i]) and
                  below_weekly_ema and
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches weekly EMA or reaches R3/R4
            if (close[i] >= ema_1w_aligned[i] or
                close[i] >= camarilla_3_aligned[i] or
                close[i] >= camarilla_h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches weekly EMA or reaches S3/S4
            if (close[i] <= ema_1w_aligned[i] or
                close[i] <= camarilla_4_aligned[i] or
                close[i] <= camarilla_3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_EMA50_Camarilla_Reversal_v1"
timeframe = "6h"
leverage = 1.0