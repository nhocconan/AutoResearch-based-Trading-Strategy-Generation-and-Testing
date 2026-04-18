#!/usr/bin/env python3
"""
1d_Williams_Alligator_Trend_Follow_With_Volume_Confirmation
Hypothesis: Williams Alligator (13,8,5 SMAs with 8,5,3 offsets) identifies trend direction. 
Go long when jaw < teeth < lips and price above lips with volume confirmation. 
Go short when jaw > teeth > lips and price below lips with volume confirmation.
In strong trends (ADX > 25), follow the trend; in choppy markets (ADX < 20), stay flat.
Uses daily timeframe for low trade frequency to minimize fee drag while capturing major trends.
Works in both bull and bear markets by following the trend direction with proper filters.
"""

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
    
    # Williams Alligator parameters
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_offset = 8
    teeth_offset = 5
    lips_offset = 3
    
    # Calculate SMAs for Alligator lines
    jaw_raw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth_raw = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips_raw = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Apply offsets (shift right by offset periods)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > jaw_offset:
        jaw[jaw_offset:] = jaw_raw[:-jaw_offset]
    if len(teeth_raw) > teeth_offset:
        teeth[teeth_offset:] = teeth_raw[:-teeth_offset]
    if len(lips_raw) > lips_offset:
        lips[lips_offset:] = lips_raw[:-lips_offset]
    
    # ADX for trend strength filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0
                
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nansum(tr[1:period+1]) if not np.any(np.isnan(tr[1:period+1])) else np.nan
        for i in range(period+1, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            else:
                atr[i] = np.nan
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(len(high)):
            if not np.isnan(atr[i]) and atr[i] > 0:
                plus_di[i] = (np.nansum(plus_dm[max(0, i-period+1):i+1]) / atr[i]) * 100
                minus_di[i] = (np.nansum(minus_dm[max(0, i-period+1):i+1]) / atr[i]) * 100
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        adx = np.zeros_like(high)
        for i in range(len(dx)):
            if i >= 2*period-1:
                valid_dx = dx[max(period, i-period+1):i+1]
                valid_dx = valid_dx[~np.isnan(valid_dx)]
                if len(valid_dx) > 0:
                    adx[i] = np.mean(valid_dx)
                else:
                    adx[i] = np.nan
            else:
                adx[i] = np.nan
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # 1-week EMA34 trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, jaw_period + jaw_offset, teeth_period + teeth_offset, lips_period + lips_offset, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        adx_val = adx[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Alligator alignment conditions
        bullish_alignment = jaw_val < teeth_val < lips_val
        bearish_alignment = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Look for trend entry with volume confirmation
            if bullish_alignment and price > lips_val and adx_val > 25 and vol_conf and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and price < lips_val and adx_val > 25 and vol_conf and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit conditions: trend weakening or reversal
            if not bullish_alignment or price < lips_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit conditions: trend weakening or reversal
            if not bearish_alignment or price > lips_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Williams_Alligator_Trend_Follow_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0