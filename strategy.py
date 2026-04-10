#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - 1d Camarilla pivot levels (R3, R4, S3, S4) on 6h chart
# - Long: price breaks above R4 with 1w bullish trend (price > 1w EMA200) and volume > 1.5x 20-period average
# - Short: price breaks below S4 with 1w bearish trend (price < 1w EMA200) and volume > 1.5x 20-period average
# - Exit: price retreats to R3 (for longs) or S3 (for shorts) OR volume drops below average
# - Target: 12-30 trades/year on 6h (50-120 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in ranging markets; breakouts at R4/S4 with trend/volume filter capture strong moves

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = pivot + range_1d * 1.1 / 2.0
    r4 = pivot + range_1d * 1.1
    s3 = pivot - range_1d * 1.1 / 2.0
    s4 = pivot - range_1d * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute 6h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup for 1w EMA200
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Price levels
        close_price = prices['close'].values[i]
        
        # Breakout conditions
        breakout_long = close_price > r4_aligned[i]
        breakout_short = close_price < s4_aligned[i]
        
        # 1w trend filter: price > EMA200 = bullish, price < EMA200 = bearish
        bullish_trend = close_price > trend_aligned[i]
        bearish_trend = close_price < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: break above R4 AND bullish trend AND volume spike
            if breakout_long and bullish_trend and volume_spike:
                position = 1
                signals[i] = 0.25
            # Short conditions: break below S4 AND bearish trend AND volume spike
            elif breakout_short and bearish_trend and volume_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price retreats to R3/S3 OR volume drops below average
            vol_exhaustion = volume[i] < vol_ma_20[i]
            exit_long = close_price < r3_aligned[i]
            exit_short = close_price > s3_aligned[i]
            exit_condition = (position == 1 and (exit_long or vol_exhaustion)) or \
                            (position == -1 and (exit_short or vol_exhaustion))
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals