#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA(34) for trend filter (bull/bear regime) and Camarilla pivot levels from prior 1d.
- Entry: Long when price breaks above R3 with bullish trend and volume > 2.0 * 6h volume MA(20);
         Short when price breaks below S3 with bearish trend and volume > 2.0 * 6h volume MA(20).
- Exit: Opposite Camarilla breakout (R3/S3) or time-based exit after 3 bars.
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels provide intraday structure; EMA filter avoids counter-trend trades; volume spike confirms conviction.
- Works in bull (breakouts with trend) and bear (fades at extremes with trend alignment).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close']
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_trend = np.zeros(len(df_1d))
    ema_trend[:] = np.nan
    ema_trend[34:] = np.where(close_1d.iloc[34:].values > ema_34_1d[34:], 1, -1)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_r3[:] = np.nan
    camarilla_s3[:] = np.nan
    
    for i in range(1, len(df_1d)):
        # Use prior day's OHLC to avoid look-ahead
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d_arr[i-1]
        camarilla_r3[i] = c + ((h - l) * 1.1 / 4)
        camarilla_s3[i] = c - ((h - l) * 1.1 / 4)
    
    # Align 1d indicators to 6h timeframe
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = volume
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20, 1)  # EMA needs 35, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_trend_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                bars_since_entry += 1
                if bars_since_entry >= 3:  # time-based exit
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: 2.0x threshold (tight to reduce trades)
        vol_spike = curr_volume > 2.0 * vol_ma_6h[i]
        
        # Trend filter: EMA direction
        bull_trend = ema_trend_aligned[i] == 1
        bear_trend = ema_trend_aligned[i] == -1
        
        if position == 0:
            bars_since_entry = 0
            # Check for entry signals
            # Long: price breaks above Camarilla R3 in bull trend with volume spike
            if curr_close > camarilla_r3_aligned[i] and bull_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in bear trend with volume spike
            elif curr_close < camarilla_s3_aligned[i] and bear_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            bars_since_entry += 1
            # Long position: exit on opposite breakout or time
            if curr_close < camarilla_s3_aligned[i] or bars_since_entry >= 3:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            bars_since_entry += 1
            # Short position: exit on opposite breakout or time
            if curr_close > camarilla_r3_aligned[i] or bars_since_entry >= 3:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0