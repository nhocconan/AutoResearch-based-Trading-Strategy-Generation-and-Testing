#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Primary signal: Price breaks above/below Camarilla R4/S4 levels from 1d
# - Volume filter: 1d volume > 1.5x 20-period average volume (ensures strong participation)
# - Trend filter: 1w close > 1w EMA20 for longs, < EMA20 for shorts (avoids counter-trend)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: Price retracement to Camarilla R3/S3 levels
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_1d_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_h4 = pivot + range_1d * 1.1 / 2  # R4
    camarilla_l4 = pivot - range_1d * 1.1 / 2  # S4
    camarilla_h3 = pivot + range_1d * 1.1 / 4  # R3
    camarilla_l3 = pivot - range_1d * 1.1 / 4  # S3
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Pre-compute 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend = close_1w > ema_20
    downtrend = close_1w < ema_20
    
    # Align HTF indicators to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    uptrend_aligned = align_htf_to_ltf(prices, df_1w, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1w, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price retracement to R3 (mean reversion) or stoploss
            if prices['close'].iloc[i] <= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price retracement to S3 (mean reversion) or stoploss
            if prices['close'].iloc[i] >= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume and trend filters
            if vol_spike_aligned[i]:
                # Long: price breaks above R4 with uptrend on 1w
                if (prices['close'].iloc[i] > camarilla_h4_aligned[i] and 
                    uptrend_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short: price breaks below S4 with downtrend on 1w
                elif (prices['close'].iloc[i] < camarilla_l4_aligned[i] and 
                      downtrend_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals