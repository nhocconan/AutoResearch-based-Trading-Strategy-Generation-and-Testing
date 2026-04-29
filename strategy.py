#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions on 6h timeframe
# Extreme readings below -80 (oversold) or above -20 (overbought) signal potential reversals
# 1d ADX > 25 confirms strong trend direction to avoid counter-trend trades in choppy markets
# Volume spike (>1.8x 20-period average) ensures institutional participation
# Works in bull/bear: ADX filter ensures we only trade with strong trends, Williams %R catches pullbacks/extensions
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 6h timeframe (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_6h = calculate_williams_r(high, low, close, 14)
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = -pd.Series(df_1d['low']).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for Williams %R (14+20 for smoothing) and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(wr_6h[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wr = wr_6h[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and strong trend (ADX > 25)
            if curr_volume_confirm and curr_adx > 25:
                # Bullish entry: Williams %R oversold (< -80) turning up
                if curr_wr < -80 and i > start_idx and wr_6h[i-1] < -80 and wr_6h[i] > wr_6h[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R overbought (> -20) turning down
                elif curr_wr > -20 and i > start_idx and wr_6h[i-1] > -20 and wr_6h[i] < wr_6h[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R reaches overbought (> -20) or trend weakens (ADX < 20)
            if curr_wr > -20 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R reaches oversold (< -80) or trend weakens (ADX < 20)
            if curr_wr < -80 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals