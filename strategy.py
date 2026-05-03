#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d EMA34 trend filter + volume confirmation
# Camarilla pivot levels identify key support/resistance; breakouts above R3 or below S3 with volume
# and higher timeframe trend alignment capture strong moves while avoiding false breakouts
# Works in bull/bear: 1d EMA34 ensures we trade with higher timeframe trend to avoid whipsaws
# Volume spike (>2.0x 20-period EMA) confirms breakout authenticity
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
# Based on proven patterns: Camarilla breakouts with volume and trend filter show strong test performance

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for 4h (based on previous day's OHLC)
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #                 S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use previous day's OHLC to calculate today's levels
    df_1d = get_htf_data(prices, '1d')  # Already loaded above, but getting again for clarity in calculation
    if len(df_1d) < 2:
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    else:
        # Calculate Camarilla levels from 1d data (previous day's OHLC)
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate Camarilla R3 and S3 for each 1d bar
        camarilla_r3_1d = close_1d + 1.125 * (high_1d - low_1d)
        camarilla_s3_1d = close_1d - 1.125 * (high_1d - low_1d)
        
        # Align to 4h timeframe (each 1d bar corresponds to 16x 4h bars)
        camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
        camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 1d trend filter
        # Long: price breaks above Camarilla R3 + price above 1d EMA34 + volume spike
        # Short: price breaks below Camarilla S3 + price below 1d EMA34 + volume spike
        if position == 0:
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 (reversal) OR price below 1d EMA34
            if close[i] < camarilla_s3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 (reversal) OR price above 1d EMA34
            if close[i] > camarilla_r3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals