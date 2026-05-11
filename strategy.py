#!/usr/bin/env python3
name = "6h_AnchoredVWAP_VolumeSpike_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for trend filter and VWAP anchoring
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d trend: EMA 34
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Pre-calculate hour for session filter (UTC 8-20)
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0
    
    # VWAP calculation variables
    vwap = np.zeros(n)
    vwap_sum = 0.0
    vol_sum = 0.0
    day_start_idx = 0
    
    for i in range(n):
        # New day detection (based on date change)
        if i == 0 or pd.Timestamp(open_time[i]).date() != pd.Timestamp(open_time[i-1]).date():
            day_start_idx = i
            vwap_sum = 0.0
            vol_sum = 0.0
        
        # Typical price
        typical_price = (high[i] + low[i] + close[i]) / 3.0
        
        # Update VWAP cumulative sums
        vwap_sum += typical_price * volume[i]
        vol_sum += volume[i]
        
        # Calculate VWAP
        if vol_sum > 0:
            vwap[i] = vwap_sum / vol_sum
        else:
            vwap[i] = typical_price
        
        # Skip until we have enough data
        if i < 20:
            continue
        
        # Volume filter: 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_filter = volume[i] > vol_ma * 1.5
        else:
            vol_filter = False
        
        in_session = (8 <= hours[i] <= 20)
        
        if not np.isnan(ema_1d_aligned[i]):
            if position == 0:
                # Long: price above VWAP + volume spike + in session + above 1d EMA
                if close[i] > vwap[i] and vol_filter and in_session and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price below VWAP + volume spike + in session + below 1d EMA
                elif close[i] < vwap[i] and vol_filter and in_session and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price below VWAP or below 1d EMA
                if close[i] < vwap[i] or close[i] < ema_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price above VWAP or above 1d EMA
                if close[i] > vwap[i] or close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals