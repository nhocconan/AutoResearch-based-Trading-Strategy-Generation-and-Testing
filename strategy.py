#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike
# Long when price breaks above upper Donchian channel in uptrend (close > 1w EMA50) with volume spike (>2x 20-period avg)
# Short when price breaks below lower Donchian channel in downtrend (close < 1w EMA50) with volume spike
# Exit when price retraces to midpoint of Donchian channel or trend reverses
# Designed for low trade frequency (~10-30/year) to minimize fee drain. Works in bull/bear by
# combining structure-based breakouts with trend filtering and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    upper_dc_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    mid_dc_1d = (upper_dc_1d + lower_dc_1d) / 2.0
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d and 1w data to 1d timeframe (no alignment needed for 1d channels)
    upper_dc_aligned = upper_dc_1d
    lower_dc_aligned = lower_dc_1d
    mid_dc_aligned = mid_dc_1d
    
    # Align 1w EMA to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_dc_aligned[i]) or 
            np.isnan(lower_dc_aligned[i]) or 
            np.isnan(mid_dc_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_val = upper_dc_aligned[i]
        lower_val = lower_dc_aligned[i]
        mid_val = mid_dc_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper DC + uptrend + volume spike
            if price > upper_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower DC + downtrend + volume spike
            elif price < lower_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price retraces to midpoint or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price touches or crosses midpoint or trend turns down
                if price <= mid_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price touches or crosses midpoint or trend turns up
                if price >= mid_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0