# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for overbought/oversold conditions and 1w ADX for trend strength
# - Uses 1w ADX > 25 to identify trending markets and < 20 for ranging markets
# - Uses 1d Williams %R to identify extreme momentum: short when > -20 (overbought), long when < -80 (oversold)
# - Enters long when Williams %R crosses above -80 from below in a ranging market (mean reversion)
# - Enters short when Williams %R crosses below -20 from above in a ranging market (mean reversion)
# - In trending markets (ADX > 25), follows the trend: long when Williams %R > -50, short when < -50
# - Designed to capture mean reversion in ranging markets and trend continuation in trending markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_1dWilliamsR_1wADX_MeanReversion_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Williams %R = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -100 * (highest_high - close_1d) / (highest_high - lowest_low),
        -50  # Neutral when no range
    )
    
    # Calculate 1w ADX (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Align 1w ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Moderate volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(adx_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r_6h[i]
        adx_val = adx_6h[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Ranging market: ADX < 20 - mean reversion at extremes
            if adx_val < 20 and vol_spike:
                # Long: Williams %R crosses above -80 from below (oversold)
                if i > 0 and williams_r_val > -80 and williams_r_6h[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above (overbought)
                elif i > 0 and williams_r_val < -20 and williams_r_6h[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
            # Trending market: ADX > 25 - follow momentum
            elif adx_val > 25 and vol_spike:
                # Long: Williams %R > -50 (bullish momentum)
                if williams_r_val > -50:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R < -50 (bearish momentum)
                elif williams_r_val < -50:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) or overbought in ranging market
            if (adx_val < 20 and williams_r_val > -20) or (adx_val >= 20 and williams_r_val < -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) or oversold in ranging market
            if (adx_val < 20 and williams_r_val < -80) or (adx_val >= 20 and williams_r_val > -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals