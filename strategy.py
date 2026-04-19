#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with daily VWAP and Bollinger Bands for mean reversion
# - Long when price touches lower Bollinger Band (20,2) and is below daily VWAP (oversold)
# - Short when price touches upper Bollinger Band (20,2) and is above daily VWAP (overbought)
# - Exit when price crosses back to daily VWAP or reaches opposite band
# - Volume filter: require volume > 1.5x 20-period average to confirm momentum
# - Works in both bull/bear markets by fading extremes at Bollinger Bands with VWAP as dynamic support/resistance
# - Target: 20-50 trades/year (80-200 total over 4 years)
name = "4h_VWAP_Bollinger_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP and Bollinger Bands (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP: cumulative (price * volume) / cumulative volume
    # Typical price = (high + low + close) / 3
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pv_1d = typical_price_1d * volume_1d
    cum_pv_1d = np.cumsum(pv_1d)
    cum_vol_1d = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_pv_1d, cum_vol_1d, out=np.full_like(cum_pv_1d, np.nan), where=cum_vol_1d!=0)
    
    # Calculate daily Bollinger Bands (20,2)
    close_ma_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    close_std_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = close_ma_1d + (2 * close_std_1d)
    lower_bb_1d = close_ma_1d - (2 * close_std_1d)
    
    # Align daily indicators to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    close_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, close_ma_1d)
    
    # Volume filter on 4h: volume > 1.5 * 20-period average
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma_4h * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or np.isnan(close_ma_1d_aligned[i]) or
            np.isnan(volume_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at or below lower BB AND below daily VWAP (oversold)
            if close[i] <= lower_bb_1d_aligned[i] and close[i] < vwap_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at or above upper BB AND above daily VWAP (overbought)
            elif close[i] >= upper_bb_1d_aligned[i] and close[i] > vwap_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses back to VWAP or reaches upper BB
            if close[i] >= vwap_1d_aligned[i] or close[i] >= upper_bb_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses back to VWAP or reaches lower BB
            if close[i] <= vwap_1d_aligned[i] or close[i] <= lower_bb_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals