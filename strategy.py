#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Volume Weighted Average Price (VWAP) and 1-week Exponential Moving Average (EMA) trend filter.
# Long when price crosses above 1d VWAP with 1w EMA uptrend and volume confirmation (>1.5x 20-period average).
# Short when price crosses below 1d VWAP with 1w EMA downtrend and volume confirmation.
# Exit when price returns to 1d VWAP or trend weakens (EMA slope flips).
# VWAP provides dynamic support/resistance based on volume, effective in both trending and ranging markets.
# Target: 20-25 trades/year per symbol (80-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Typical Price and VWAP for 1d
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    # Handle division by zero at start
    vwap_1d[vwap_denominator == 0] = np.nan
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(21) on 1w
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    # EMA slope: current EMA - previous EMA
    ema_slope = np.diff(ema_1w, prepend=np.nan)
    
    # Align indicators to lower timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need VWAP and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: EMA slope positive for uptrend, negative for downtrend
        uptrend = ema_slope_aligned[i] > 0
        downtrend = ema_slope_aligned[i] < 0
        
        if position == 0:
            # Look for VWAP crossovers
            # Long: price crosses above VWAP AND uptrend
            if (close[i] > vwap_1d_aligned[i] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price crosses below VWAP AND downtrend
            elif (close[i] < vwap_1d_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or trend weakens (EMA slope <= 0)
            if (close[i] <= vwap_1d_aligned[i] or 
                ema_slope_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP or trend weakens (EMA slope >= 0)
            if (close[i] >= vwap_1d_aligned[i] or 
                ema_slope_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VWAP_1wEMA_Trend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0