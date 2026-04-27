# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels from daily data provide strong support/resistance.
# At 6h timeframe, price breaking above R3 or below S3 with volume confirmation and daily trend alignment
# indicates momentum continuation. This captures breakout moves while avoiding false breakouts in ranging markets.
# Works in both bull and bear markets by following daily trend direction.
# Target: 15-35 trades/year per symbol to minimize fee drag.

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
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using standard Camarilla formulas based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = close_1d + (range_1d * 1.1000 / 4)  # ~1.1/4 = 0.275
    s3 = close_1d - (range_1d * 1.1000 / 4)
    r4 = close_1d + (range_1d * 1.1000 / 2)  # ~1.1/2 = 0.55
    s4 = close_1d - (range_1d * 1.1000 / 2)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume spike detector (volume > 2x 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_spike = vol_1d > (2.0 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla levels, trend, and volume spike
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 6h price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        r3_now = r3_6h[i]
        s3_now = s3_6h[i]
        r4_now = r4_6h[i]
        s4_now = s4_6h[i]
        trend_now = ema_34_1d_aligned[i]
        vol_spike_now = vol_spike_aligned[i] > 0.5  # Boolean from aligned spike
        
        # Breakout conditions
        breakout_up = price_now > r3_now
        breakout_down = price_now < s3_now
        
        # Entry conditions
        if position == 0:
            # Long: break above R3 with volume spike and daily uptrend
            if breakout_up and vol_spike_now and price_now > trend_now:
                signals[i] = size
                position = 1
            # Short: break below S3 with volume spike and daily downtrend
            elif breakout_down and vol_spike_now and price_now < trend_now:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S3 or reverses below R3
            if price_now < s3_now or price_now < r3_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 or reverses above S3
            if price_now > r3_now or price_now > s3_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0