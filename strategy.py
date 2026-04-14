# 4h_1d_PriceChannel_Breakout_Volume_Trend_v1
# Hypothesis: On 4h timeframe, use 1d price channel breakout with volume confirmation and ADX trend filter.
# Breakout above 1d high or below 1d low with volume > 2x 20-period average and ADX > 25 indicates strong trend.
# Go long on breakout above 1d high, short on breakout below 1d low.
# Exit when price crosses 1d VWAP or ADX falls below 20 indicating trend exhaustion.
# Designed to capture strong trends in both bull and bear markets while avoiding false breakouts in low-volume, low-volatility environments.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for price channel and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = np.full_like(close_1d, np.nan)
    cum_vol = np.cumsum(volume_1d)
    cum_tpv = np.cumsum(typical_price_1d * volume_1d)
    vwap_1d = np.where(cum_vol > 0, cum_tpv / cum_vol, np.nan)
    
    # Align 1d high, low, and VWAP to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate ADX on 1d data
    if len(high_1d) < 14:
        return np.zeros(n)
    
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]):
            continue
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - high_1d[i-1]), 
                   abs(low_1d[i] - low_1d[i-1]))
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/14)
    atr = np.zeros_like(high_1d)
    plus_di = np.zeros_like(high_1d)
    minus_di = np.zeros_like(high_1d)
    dx = np.zeros_like(high_1d)
    adx = np.full_like(high_1d, np.nan)
    
    if len(high_1d) >= 14:
        # Initial values
        atr[13] = np.nansum(tr[1:14])
        plus_dm_sum = np.nansum(plus_dm[1:14])
        minus_dm_sum = np.nansum(minus_dm[1:14])
        
        for i in range(14, len(high_1d)):
            if np.isnan(tr[i]) or np.isnan(plus_dm[i]) or np.isnan(minus_dm[i]):
                atr[i] = atr[i-1]
                plus_dm_sum = plus_dm_sum
                minus_dm_sum = minus_dm_sum
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                plus_dm_sum = (plus_dm_sum * 13 + plus_dm[i]) / 14
                minus_dm_sum = (minus_dm_sum * 13 + minus_dm[i]) / 14
            
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_sum / atr[i]
                minus_di[i] = 100 * minus_dm_sum / atr[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Calculate ADX as smoothed DX
        if len(high_1d) >= 27:
            adx[26] = np.nanmean(dx[14:27])
            for i in range(27, len(high_1d)):
                if np.isnan(dx[i]):
                    adx[i] = adx[i-1]
                else:
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after enough data for alignment
        # Skip if any critical data is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        vol_ma_20_aligned = vol_ma_20  # Already on 4h timeframe
        
        if np.isnan(vol_ma_20_aligned[i]) or vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for long entries: breakout above 1d high with volume surge in strong trend
            if (close[i] > high_1d_aligned[i] and 
                volume_ratio > 2.0 and
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Look for short entries: breakdown below 1d low with volume surge in strong trend
            elif (close[i] < low_1d_aligned[i] and 
                  volume_ratio > 2.0 and
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d VWAP or ADX falls indicating trend exhaustion
            if (close[i] < vwap_1d_aligned[i] or
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d VWAP or ADX falls indicating trend exhaustion
            if (close[i] > vwap_1d_aligned[i] or
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_PriceChannel_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0