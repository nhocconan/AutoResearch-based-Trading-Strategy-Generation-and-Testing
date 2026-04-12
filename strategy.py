#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
    # Uses 1d EMA200 for trend filter: only take breakouts in direction of 1d trend
    # Camarilla levels from 1d: long at R3/S3 breakout, short at S3/R3 breakdown
    # Volume confirmation: volume > 2.0 * 20-period average to filter false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-30 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_R4 = np.full(n, np.nan)
    camarilla_S4 = np.full(n, np.nan)
    
    # Pivot point = (H + L + C) / 3
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    # R4 = Close + (High - Low) * 1.1
    # S4 = Close - (High - Low) * 1.1
    for i in range(1, n):
        phigh = high_1d[i-1] if i-1 < len(high_1d) else high_1d[-1]
        plow = low_1d[i-1] if i-1 < len(low_1d) else low_1d[-1]
        pclose = close_1d[i-1] if i-1 < len(close_1d) else close_1d[-1]
        
        camarilla_R3[i] = pclose + (phigh - plow) * 1.1 / 2
        camarilla_S3[i] = pclose - (phigh - plow) * 1.1 / 2
        camarilla_R4[i] = pclose + (phigh - plow) * 1.1
        camarilla_S4[i] = pclose - (phigh - plow) * 1.1
    
    # Align Camarilla levels to 6h timeframe (already aligned by get_htf_data + loop index)
    # No additional alignment needed as we're using previous day's levels
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Camarilla R3 in bullish trend
        if bullish_trend:
            long_entry = (close[i] > camarilla_R3[i]) and volume_spike[i]
        # Short breakout: price breaks below Camarilla S3 in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_S3[i]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_S3[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > camarilla_R3[i]) or (not bullish_trend and not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0