# 4h_Camarilla_Reversal_ADX_Volume
# Strategy: Fades extreme levels (S3/R3) in ranging markets, breaks S4/R4 in trending markets
# Uses 1d ADX for trend filter and volume confirmation to reduce false signals
# Designed for 20-30 trades/year to minimize fee drag
# Works in both bull and bear markets by adapting to regime

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h4 = c + range_val * 1.1 / 2
    l4 = c - range_val * 1.1 / 2
    h3 = c + range_val * 1.1 / 4
    l3 = c - range_val * 1.1 / 4
    h2 = c + range_val * 1.1 / 6
    l2 = c - range_val * 1.1 / 6
    h1 = c + range_val * 1.1 / 12
    l1 = c - range_val * 1.1 / 12
    return h4, l4, h3, l3, h2, l2, h1, l1

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        atr = np.full(n, np.nan)
        plus_dm_smooth = np.full(n, np.nan)
        minus_dm_smooth = np.full(n, np.nan)
        
        # Initial values
        if n >= period:
            atr[period-1] = np.nanmean(tr[1:period+1])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period+1])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period+1])
            
            # Wilder smoothing
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX
        adx = np.full(n, np.nan)
        if n >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need Camarilla calculation and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 4h bar
        h4, l4, h3, l3, h2, l2, h1, l1 = calculate_camarilla(high[i], low[i], close[i-1])
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: ADX threshold
        trending = adx_1d_4h[i] >= 25
        ranging = adx_1d_4h[i] < 25
        
        if position == 0:
            # Ranging market: fade extremes (S3/R3)
            if ranging:
                # Long near S3 with volume confirmation
                if close[i] <= l3 * 1.001 and vol_confirmed:  # Allow small buffer
                    signals[i] = 0.25
                    position = 1
                # Short near R3 with volume confirmation
                elif close[i] >= h3 * 0.999 and vol_confirmed:  # Allow small buffer
                    signals[i] = -0.25
                    position = -1
            # Trending market: breakout of S4/R4
            else:
                # Long breakout above R4 with volume
                if close[i] > h4 and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S4 with volume
                elif close[i] < l4 and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: mean reversion to pivot or stop
            if close[i] >= (h3 + l3) / 2:  # Return to midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: mean reversion to pivot or stop
            if close[i] <= (h3 + l3) / 2:  # Return to midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Reversal_ADX_Volume"
timeframe = "4h"
leverage = 1.0