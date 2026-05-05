#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with volume confirmation and 1d EMA34 trend filter
# Long when price breaks above Camarilla R3 AND volume > 1.5x 20-period average AND 1d EMA34 uptrend
# Short when price breaks below Camarilla S3 AND volume > 1.5x 20-period average AND 1d EMA34 downtrend
# Exit when price crosses Camarilla pivot (PP) OR 1d trend reverses
# Uses discrete sizing (0.30) to limit fee drag. Target: 25-40 trades/year per symbol.
# Camarilla levels provide institutional support/resistance, volume confirms breakout strength,
# 1d EMA34 ensures alignment with higher timeframe direction to avoid whipsaws.

name = "4h_Camarilla_R3S3_VolumeConfirm_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h data (using previous day's OHLC)
    # Camarilla formula: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate PP, R3, S3 for each 4h bar using previous bar's data
    pp_4h = np.zeros_like(high_4h)
    r3_4h = np.zeros_like(high_4h)
    s3_4h = np.zeros_like(high_4h)
    
    for i in range(1, len(df_4h)):
        # Use previous bar's OHLC to calculate current bar's Camarilla levels
        prev_high = high_4h[i-1]
        prev_low = low_4h[i-1]
        prev_close = close_4h[i-1]
        
        pp = (prev_high + prev_low + prev_close) / 3.0
        r3 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
        s3 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
        
        pp_4h[i] = pp
        r3_4h[i] = r3
        s3_4h[i] = s3
    
    # For first bar, use same values (will be aligned properly)
    pp_4h[0] = pp_4h[1] if len(pp_4h) > 1 else pp_4h[0]
    r3_4h[0] = r3_4h[1] if len(r3_4h) > 1 else r3_4h[0]
    s3_4h[0] = s3_4h[1] if len(s3_4h) > 1 else s3_4h[0]
    
    # Align Camarilla levels to prices timeframe
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND volume confirmation AND 1d EMA34 uptrend
            if (close[i] > r3_aligned[i] and 
                volume_filter[i] and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below S3 AND volume confirmation AND 1d EMA34 downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_filter[i] and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below PP OR 1d trend changes to downtrend
            if (close[i] < pp_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above PP OR 1d trend changes to uptrend
            if (close[i] > pp_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals