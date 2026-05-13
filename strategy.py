#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 with volume > 1.5x 20-period average AND close > 1d EMA34.
# Short when price breaks below S3 with volume > 1.5x 20-period average AND close < 1d EMA34.
# Exit when price retouches the Camarilla pivot point (PP) or volume drops below average.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness
# by trading institutional breakout levels with trend and volume filters to avoid false signals.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback for volume MA
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for previous 1d bar (using HTF data)
        # Need at least 1 day of data to calculate levels
        if i < len(prices):
            # Get the most recent completed 1d bar for Camarilla calculation
            # We'll use the 1d data we already fetched
            idx_1d = min(len(df_1d) - 1, i // 24)  # approximate 1d bar index (24*4h bars per day)
            if idx_1d < 1:
                signals[i] = 0.0
                continue
                
            # Use previous completed 1d bar (idx_1d-1) to avoid look-ahead
            lookback_idx = idx_1d - 1
            if lookback_idx < 0:
                signals[i] = 0.0
                continue
                
            high_1d = df_1d['high'].values[lookback_idx]
            low_1d = df_1d['low'].values[lookback_idx]
            close_1d_val = df_1d['close'].values[lookback_idx]
            
            # Calculate Camarilla levels
            range_1d = high_1d - low_1d
            if range_1d <= 0:
                signals[i] = 0.0
                continue
                
            # Camarilla R3, S3, and PP (pivot point)
            r3 = close_1d_val + range_1d * 1.1 / 4
            s3 = close_1d_val - range_1d * 1.1 / 4
            pp = (high_1d + low_1d + close_1d_val) / 3
            
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirm = volume[i] > 1.5 * volume_ma[i]
            
            if position == 0:
                # LONG: price breaks above R3 with volume confirmation AND close > 1d EMA34 (uptrend)
                if (close[i] > r3 and 
                    volume_confirm and 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # SHORT: price breaks below S3 with volume confirmation AND close < 1d EMA34 (downtrend)
                elif (close[i] < s3 and 
                      volume_confirm and 
                      close[i] < ema_34_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: price retouches pivot point OR volume drops below average
                if (close[i] <= pp or 
                    volume[i] < volume_ma[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: price retouches pivot point OR volume drops below average
                if (close[i] >= pp or 
                    volume[i] < volume_ma[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals