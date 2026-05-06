#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d ADX trend filter with 12h Bollinger Band squeeze breakout
# - Uses 1d ADX > 25 to identify trending markets (avoids chop)
# - Uses 12h Bollinger Bands (20,2) squeeze detection for low volatility periods
# - Enters long when price breaks above 12h upper BB with volume confirmation
# - Enters short when price breaks below 12h lower BB with volume confirmation
# - Bollinger Band squeeze (BB width < 50th percentile) indicates imminent volatility expansion
# - Volume confirmation (1.5x 20-period MA) ensures institutional participation
# - Designed to catch volatility breakouts after consolidation in trending markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "4h_1dADX_12hBB_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr = np.zeros(len(high_1d))
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
    
    # Wilder's smoothing
    atr_1d = np.zeros(len(high_1d))
    plus_di_1d = np.zeros(len(high_1d))
    minus_di_1d = np.zeros(len(high_1d))
    
    atr_1d[13] = np.mean(tr[1:15])
    plus_dm_sum = np.sum(plus_dm[1:15])
    minus_dm_sum = np.sum(minus_dm[1:15])
    
    for i in range(15, len(high_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
        plus_dm_sum = plus_dm_sum - plus_dm[i-14] + plus_dm[i]
        minus_dm_sum = minus_dm_sum - minus_dm[i-14] + minus_dm[i]
        plus_di_1d[i] = 100 * plus_dm_sum / (atr_1d[i] * 14) if atr_1d[i] != 0 else 0
        minus_di_1d[i] = 100 * minus_dm_sum / (atr_1d[i] * 14) if atr_1d[i] != 0 else 0
    
    # DX and ADX
    dx_1d = np.zeros(len(high_1d))
    adx_1d = np.zeros(len(high_1d))
    
    for i in range(27, len(high_1d)):
        di_diff = abs(plus_di_1d[i] - minus_di_1d[i])
        di_sum = plus_di_1d[i] + minus_di_1d[i]
        dx_1d[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
    
    adx_1d[27] = np.mean(dx_1d[27:41]) if 41 <= len(high_1d) else 0
    for i in range(41, len(high_1d)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14
    
    # Align 1d ADX to 4h
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    adx_filter = adx_1d_4h > 25  # Strong trend filter
    
    # Get 12h data for Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band squeeze: width < 50th percentile of last 50 periods
    bb_width_percentile = np.zeros_like(bb_width)
    for i in range(50, len(bb_width)):
        bb_width_percentile[i] = np.percentile(bb_bb_width[i-50:i], 50) if i >= 50 else bb_width[i]
    
    squeeze = bb_width < bb_width_percentile  # Low volatility condition
    
    # Align 12h BB and squeeze to 4h
    upper_bb_4h = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_4h = align_htf_to_ltf(prices, df_12h, lower_bb)
    squeeze_4h = align_htf_to_ltf(prices, df_12h, squeeze)
    
    # Volume filter (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_bb_4h[i]) or np.isnan(lower_bb_4h[i]) or 
            np.isnan(squeeze_4h[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze breakout above upper band with volume and trend
            if squeeze_4h[i] and close[i] > upper_bb_4h[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower band with volume and trend
            elif squeeze_4h[i] and close[i] < lower_bb_4h[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle of BB or opposite band
            middle_bb = (upper_bb_4h[i] + lower_bb_4h[i]) / 2
            if close[i] < middle_bb or close[i] < lower_bb_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle of BB or opposite band
            middle_bb = (upper_bb_4h[i] + lower_bb_4h[i]) / 2
            if close[i] > middle_bb or close[i] > upper_bb_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals