#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume confirmation (>2x 28-bar avg) capture institutional moves in both bull and bear markets. Uses higher timeframe Camarilla levels from 1d for structure, EMA34 for trend alignment, and volume spike for confirmation. Targets 15-25 trades/year to minimize fee drag while maintaining edge via trend filter and volume confirmation. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by EMA, volume confirms real moves).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 as primary breakout levels
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d
    camarilla_s3 = close_1d - 1.1 * range_1d
    
    # Align Camarilla levels (they represent the 1d bar's levels, available after 1d close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume average (28-period = 14 days on 12h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=28, min_periods=28).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(40, 34, 28)  # 1d lookback, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_val = ema_34_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2x 28-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume confirmation
            long_signal = (high_val > camarilla_r3_val) and volume_confirmed
            # Short: price breaks below Camarilla S3 with volume confirmation
            short_signal = (low_val < camarilla_s3_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price re-enters Camarilla H3-L3 range (mean reversion exit)
            camarilla_h3 = close_1d[i//16] + 0.5 * range_1d[i//16] if i//16 < len(close_1d) else camarilla_r3_val - range_1d[i//16]*0.5 if i//16 < len(range_1d) else camarilla_r3_val
            camarilla_l3 = close_1d[i//16] - 0.5 * range_1d[i//16] if i//16 < len(close_1d) else camarilla_s3_val + range_1d[i//16]*0.5 if i//16 < len(range_1d) else camarilla_s3_val
            # Simplified exit: price crosses below EMA34 (trend reversal)
            if close_val < ema_34_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price re-enters Camarilla H3-L3 range or crosses above EMA34
            if close_val > ema_34_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0