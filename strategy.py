# 6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike
# Hypothesis: Breakouts at Camarilla R4/S4 levels with 1d trend filter and volume spike capture strong momentum in both bull and bear markets. R4/S4 represent extreme levels where breakouts often signal new trends, while fading at these levels is less common. Volume confirms institutional participation. Trend filter ensures alignment with higher timeframe direction. Targets 15-30 trades/year to avoid fee drag.
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
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels (R4, S4)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align daily levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.5x 24-period average (48 hours)
    vol_ma_24 = np.full(n, np.nan, dtype=np.float64)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d EMA (34 periods), daily data, volume MA (24 periods)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: volume > 2.5x average
        vol_filter = vol_now > 2.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above R4 + bullish trend + volume spike
            if price > r4_level and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S4 + bearish trend + volume spike
            elif price < s4_level and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to EMA trend or trend turns bearish
            if price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to EMA trend or trend turns bullish
            if price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0