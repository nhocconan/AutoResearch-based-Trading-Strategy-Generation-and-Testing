# 4h_Donchian20_VolumeSpike_TrendFilter_v1
# Donchian breakout strategy on 4h with volume spike confirmation and EMA trend filter
# Long when price breaks above upper band with volume spike and price > EMA50
# Short when price breaks below lower band with volume spike and price < EMA50
# Uses 4h EMA50 for trend filter, 40-period volume spike, and ATR-based stop
# Designed for fewer trades (target: 50-100/year) to minimize fee drag

#!/usr/bin/env python3
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
    
    # === 4h Donchian Channels (20-period) ===
    # Upper band: highest high over 20 periods
    upper_band = np.full_like(high, np.nan)
    # Lower band: lowest low over 20 periods
    lower_band = np.full_like(low, np.nan)
    
    for i in range(len(high)):
        if i >= 19:
            upper_band[i] = np.max(high[i-19:i+1])
            lower_band[i] = np.min(low[i-19:i+1])
        elif i > 0:
            upper_band[i] = np.max(high[0:i+1])
            lower_band[i] = np.min(low[0:i+1])
        else:
            upper_band[i] = high[i]
            lower_band[i] = low[i]
    
    # === 4h EMA50 for trend filter ===
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 40-period volume average for spike detection ===
    vol_ma_40 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 39:
            vol_ma_40[i] = np.mean(volume[i-39:i+1])
        elif i > 0:
            vol_ma_40[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma_40[i] = volume[0]
    
    # Volume spike: current volume > 2.0 x 40-period average
    vol_spike = volume > vol_ma_40 * 2.0
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper band with volume spike and uptrend (price > EMA50)
            if (close[i] > upper_band[i] and vol_spike[i] and close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower band with volume spike and downtrend (price < EMA50)
            elif (close[i] < lower_band[i] and vol_spike[i] and close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to lower band OR trend reverses (price < EMA50)
            if close[i] < lower_band[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to upper band OR trend reverses (price > EMA50)
            if close[i] > upper_band[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0