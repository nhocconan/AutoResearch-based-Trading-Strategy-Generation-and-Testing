#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 breakout with volume confirmation and daily ADX trend filter.
# Camarilla levels derived from prior 1-day high-low-close provide institutional support/resistance.
# Breakouts above R1 or below S1 with volume > 2x 20-period average signal institutional participation.
# ADX > 25 on daily timeframe filters for trending markets, avoiding false breakouts in ranges.
# Designed for 12h timeframe to capture multi-day trends with low trade frequency (~15-25/year).
# Entry: Long when price breaks above R1 with volume spike and ADX>25; Short when breaks below S1 with volume spike and ADX>25.
# Exit: Price re-enters the central zone (between S1 and R1) or ADX drops below 20.
# Uses tight conditions to limit trades and avoid overtrading.

name = "12h_Camarilla_ADX_Volume"
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
    
    # Get daily data for Camarilla pivots and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using prior day's values to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Shift by 1 to use prior day's data
    phigh = np.roll(daily_high, 1)
    plow = np.roll(daily_low, 1)
    pclose = np.roll(daily_close, 1)
    # First day has no prior - set to NaN
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Calculate Camarilla R1 and S1
    camarilla_width = (phigh - plow) * 1.1 / 12
    r1 = pclose + camarilla_width
    s1 = pclose - camarilla_width
    
    # Calculate ADX on daily timeframe
    # ADX requires +DI, -DI, and TR
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = phigh - plow
    tr2 = np.abs(phigh - pclose)
    tr3 = np.abs(plow - pclose)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = phigh - np.roll(phigh, 1)
    down_move = np.roll(plow, 1) - plow
    # First element has no prior
    up_move[0] = np.nan
    down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    # Smooth TR, +DM, -DM over 14 periods
    atr = wilders_smooth(tr, 14)
    plus_di = wilders_smooth(plus_dm, 14)
    minus_di = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    dx = np.full_like(atr, np.nan)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    # ADX is smoothed DX
    adx = wilders_smooth(dx, 14)
    
    # Align daily indicators to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 2.0 * 20-period average on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(adx_12h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and ADX>25
            if (close[i] > r1_12h[i] and 
                volume_spike[i] and 
                adx_12h[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and ADX>25
            elif (close[i] < s1_12h[i] and 
                  volume_spike[i] and 
                  adx_12h[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price re-enters below R1 OR ADX drops below 20 (trend weakening)
            if (close[i] < r1_12h[i]) or (adx_12h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price re-enters above S1 OR ADX drops below 20
            if (close[i] > s1_12h[i]) or (adx_12h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals