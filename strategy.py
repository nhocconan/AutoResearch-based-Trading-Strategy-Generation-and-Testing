#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout with 1d Volume Confirmation and 1w Trend Filter
# - Camarilla pivot levels (R3, R4, S3, S4) calculated from 1d OHLC
# - Long when price breaks above R4 with volume > 1.5x 20-bar average AND 1w close > 1w EMA50
# - Short when price breaks below S3 with volume > 1.5x 20-bar average AND 1w close < 1w EMA50
# - Exit when price returns to the 1d close (pivot point) or opposite level is touched
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in ranging markets and catch breakouts in trends
# - Volume confirmation filters false breakouts
# - 1w trend filter ensures we trade with higher timeframe momentum

name = "6h_1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
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
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP) and Camarilla levels
    # PP = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    rng = high_1d - low_1d
    # Camarilla levels
    r3 = pp + rng * 1.1 / 4.0
    r4 = pp + rng * 1.1 / 2.0
    s3 = pp - rng * 1.1 / 4.0
    s4 = pp - rng * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above R4 with volume spike and 1w uptrend
            if (prices['close'].iloc[i] > r4_aligned[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below S3 with volume spike and 1w downtrend
            elif (prices['close'].iloc[i] < s3_aligned[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to the pivot point (PP)
            # 2. Opposite level is touched (long exits at S3, short exits at R3)
            if position == 1:
                if prices['close'].iloc[i] <= pp_aligned[i] or prices['close'].iloc[i] < s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if prices['close'].iloc[i] >= pp_aligned[i] or prices['close'].iloc[i] > r3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals