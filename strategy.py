#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA34 trend filter + volume spike confirmation.
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 24-bar average.
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 24-bar average.
# Uses discrete position sizing (0.25) to limit drawdown and fee churn. Works in bull/bear via 1d EMA34 trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    camarilla_r3 = close_1d_arr + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d_arr - 1.1 * (high_1d - low_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*5m = 2h for 12h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 + uptrend + volume spike
            if curr_high > camarilla_r3_aligned[i] and curr_close > ema_34_1d_aligned[i] and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 + downtrend + volume spike
            elif curr_low < camarilla_s3_aligned[i] and curr_close < ema_34_1d_aligned[i] and curr_volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price closes below EMA34 (trend change)
            if curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above EMA34 (trend change)
            if curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals