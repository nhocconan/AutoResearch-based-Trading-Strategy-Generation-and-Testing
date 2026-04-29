#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike filter and EMA50 trend
# Long when price breaks above R3 AND 1d volume > 2x 20-bar average AND price > 1d EMA50
# Short when price breaks below S3 AND 1d volume > 2x 20-bar average AND price < 1d EMA50
# Exit when price retouches the central pivot (PP) level
# Uses discrete sizing (0.25) to limit fee drag. Target: 25-40 trades/year on 4h.
# Camarilla levels provide high-probability reversal points; volume confirms conviction.
# 1d EMA50 ensures alignment with higher timeframe trend, reducing counter-trend trades.

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume spike filter (>2x 20-bar average)
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    typical_price_1d_arr = typical_price_1d.values
    
    # Camarilla levels: R4 = PP + 1.5*(H-L), R3 = PP + 1.25*(H-L), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    daily_range = high_1d - low_1d
    
    # Pivot point (PP)
    pp_1d = typical_price_1d_arr
    # Resistance levels
    r3_1d = pp_1d + 1.25 * daily_range
    # Support levels
    s3_1d = pp_1d - 1.25 * daily_range
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA50 needs 50 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(pp_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        pp = pp_1d_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND volume spike AND price > 1d EMA50
            if curr_close > r3 and vol_spike and curr_close > ema_50:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND volume spike AND price < 1d EMA50
            elif curr_close < s3 and vol_spike and curr_close < ema_50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches pivot point (PP)
            if curr_close <= pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches pivot point (PP)
            if curr_close >= pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals