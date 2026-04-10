#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot + 1d volume confirmation + 1w trend filter
# - Camarilla levels calculated from prior 1d bar (H1D, L1D, C1D)
# - Long when price closes above R3 with volume > 1.5x 24-bar avg AND 1w close > 1w EMA50
# - Short when price closes below S3 with volume > 1.5x 24-bar avg AND 1w close < 1w EMA50
# - Exit when price returns to Camarilla pivot point (PP)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20 trades/year (80 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in ranging/volatile markets (2022-2025)
# - Volume confirmation filters false breakouts
# - 1w trend filter ensures alignment with higher timeframe

name = "12h_1d_1w_camarilla_pivot_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pp_1d = typical_price_1d  # PP = (H+L+C)/3
    range_1d = high_1d - low_1d
    r3_1d = pp_1d + (range_1d * 1.1 / 2.0)
    s3_1d = pp_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 12h volume confirmation: > 1.5x 24-period average (2 days)
    volume_24_avg = prices['volume'].rolling(window=24, min_periods=24).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_24_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_24_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price closes above R3 with volume spike and 1w uptrend
            if (prices['close'].iloc[i] > r3_1d_aligned[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price closes below S3 with volume spike and 1w downtrend
            elif (prices['close'].iloc[i] < s3_1d_aligned[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to Camarilla pivot point (PP)
            if position == 1 and prices['close'].iloc[i] < pp_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > pp_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals