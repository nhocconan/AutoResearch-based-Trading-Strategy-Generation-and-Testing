#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3S3 Breakout with 4h EMA34 Trend Filter and Volume Spike
- Uses 4h EMA34 for higher timeframe trend direction (long only above EMA34, short only below)
- 1h Camarilla R3/S3 levels provide precise entry points for breakouts
- Volume confirmation (> 2.0x 20-period average) filters weak breakouts
- Exit when price returns to Camarilla Pivot point or trend reverses
- Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
- Works in both bull and bear markets by trading breakouts in direction of 4h trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h Camarilla levels (using prior bar's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    # We use the prior completed bar to avoid look-ahead
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    # Set first bar's prior values to NaN (no prior bar)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    camarilla_pivot = prior_close
    camarilla_r3 = prior_close + ((prior_high - prior_low) * 1.1 / 4)
    camarilla_s3 = prior_close - ((prior_high - prior_low) * 1.1 / 4)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # for EMA34, volume MA, and prior bar data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or np.isnan(prior_close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND above 4h EMA34 AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND below 4h EMA34 AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to Camarilla Pivot OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price <= Camarilla Pivot OR price closes below 4h EMA34
                if (close[i] <= camarilla_pivot[i] or close[i] < ema_34_4h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price >= Camarilla Pivot OR price closes above 4h EMA34
                if (close[i] >= camarilla_pivot[i] or close[i] > ema_34_4h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0