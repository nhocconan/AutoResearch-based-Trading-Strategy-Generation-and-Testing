#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d ATR volatility filter and volume confirmation
- Long when price breaks above Camarilla R3 level AND ATR(14) > 1.5x ATR(50) AND volume > 1.8x 20-period average
- Short when price breaks below Camarilla S3 level AND ATR(14) > 1.5x ATR(50) AND volume > 1.8x 20-period average
- Exit when price crosses Camarilla pivot point (mean reversion to center)
- Uses 1d ATR ratio for volatility regime filter to avoid low-momentum false breakouts
- Volume confirmation threshold set to 1.8x to balance signal quality and trade frequency
- Designed for both bull and bear markets: volatility filter ensures momentum behind breakouts
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2, PP = (H+L+C)/3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate ATR(14) and ATR(50) on 1d data for volatility regime filter
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = tr.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align ATR values to 4h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # ATR ratio: short-term / long-term volatility (> 1.5 indicates elevated volatility)
    atr_ratio = atr14_aligned / atr50_aligned
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(atr_ratio[i]) or 
            np.isnan(vol_ma[i]) or
            atr50_aligned[i] == 0):  # Avoid division by zero
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Break above R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # Break below S3
        
        # Volatility regime filter (elevated short-term volatility)
        vol_filter = atr_ratio[i] > 1.5
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + volatility filter + volume confirmation
            if breakout_up and vol_filter and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down + volatility filter + volume confirmation
            elif breakout_down and vol_filter and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Camarilla pivot point (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below pivot point
                if close[i] < camarilla_pp_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above pivot point
                if close[i] > camarilla_pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dATR_VolumeFilter_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0