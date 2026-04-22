#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian channel breakout with 1w EMA50 trend filter and volume confirmation
    # Donchian(20) breakouts capture sustained trends. EMA50 on 1w filters for major trend direction.
    # Volume confirmation ensures breakouts have conviction. This combination works in both bull and bear markets.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channels (20-period) on 1d
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_channel = rolling_max(high_1d, 20)
    lower_channel = rolling_min(low_1d, 20)
    
    # 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation (20-period average on 1d)
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    vol_spike = volume_1d > 1.5 * vol_ma20_1d_aligned
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Align Donchian channels to 1d timeframe (already 1d, no alignment needed)
    # But we need to align to the lower timeframe (1d bars in prices)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper channel + above 1w EMA50 + volume spike
            if close[i] > upper_aligned[i] and close[i] > ema50_1w_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel + below 1w EMA50 + volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema50_1w_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses back to opposite channel or trend reversal
            if position == 1:
                if close[i] < lower_aligned[i] or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_aligned[i] or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0