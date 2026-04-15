# 6h_200EMA_Trend_Pullback_With_Volume_Spike
# Strategy: Buy pullbacks to 200EMA in uptrend, sell rallies to 200EMA in downtrend
# Uptrend: price above 200EMA + weekly higher high; Downtrend: price below 200EMA + weekly lower low
# Entry: Price touches 200EMA ± 0.5% with volume spike (2x 20-period avg)
# Exit: Opposite signal or trend reversal
# Works in bull/bear by following higher timeframe trend; avoids whipsaws with volume confirmation
# Target: 20-40 trades/year on 6f timeframe

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 200 EMA on 6h timeframe
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly trend filter: higher high/low for uptrend, lower high/low for downtrend
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly swing points
    weekly_hh = np.maximum.accumulate(weekly_high)  # Higher highs
    weekly_hl = np.maximum.accumulate(weekly_low)   # Higher lows
    weekly_lh = np.minimum.accumulate(weekly_high)  # Lower highs
    weekly_ll = np.minimum.accumulate(weekly_low)   # Lower lows
    
    # Align weekly trend to 6h
    weekly_hh_6h = align_htf_to_ltf(prices, weekly, weekly_hh)
    weekly_hl_6h = align_htf_to_ltf(prices, weekly, weekly_hl)
    weekly_lh_6h = align_htf_to_ltf(prices, weekly, weekly_lh)
    weekly_ll_6h = align_htf_to_ltf(prices, weekly, weekly_ll)
    
    # Volume filter: current volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Proximity to 200EMA: within 0.5%
    ema_distance = np.abs(close - ema_200) / ema_200
    near_ema = ema_distance <= 0.005
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if required data is NaN
        if (np.isnan(ema_200[i]) or np.isnan(weekly_hh_6h[i]) or 
            np.isnan(weekly_hl_6h[i]) or np.isnan(weekly_lh_6h[i]) or 
            np.isnan(weekly_ll_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend based on weekly structure
        # Uptrend: price above EMA200 and weekly making higher highs/lows
        # Downtrend: price below EMA200 and weekly making lower highs/lows
        is_uptrend = (close[i] > ema_200[i]) and (weekly_hh_6h[i] > weekly_hh_6h[i-1]) and (weekly_hl_6h[i] > weekly_hl_6h[i-1])
        is_downtrend = (close[i] < ema_200[i]) and (weekly_lh_6h[i] < weekly_lh_6h[i-1]) and (weekly_ll_6h[i] < weekly_ll_6h[i-1])
        
        # Only trade with volume spike and near EMA
        if volume_spike[i] and near_ema[i]:
            if is_uptrend:
                # Long on pullback to EMA in uptrend
                signals[i] = 0.25
            elif is_downtrend:
                # Short on rally to EMA in downtrend
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_200EMA_Trend_Pullback_With_Volume_Spike"
timeframe = "6h"
leverage = 1.0