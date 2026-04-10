#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot + 1w volume confirmation + 1d trend filter
# - Camarilla levels calculated from prior 1w bar (H1w, L1w, C1w)
# - Long when price closes above R3 with volume > 1.5x 20-bar avg AND 1d close > 1d EMA50
# - Short when price closes below S3 with volume > 1.5x 20-bar avg AND 1d close < 1d EMA50
# - Exit when price returns to Camarilla pivot point (PP)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20 trades/year (80 total over 4 years) to avoid fee drag
# - Weekly timeframe provides stronger volume confirmation for significant breakouts
# - 1d trend filter ensures alignment with higher timeframe
# - Designed for both bull and bear markets: volume confirms institutional interest, trend filter avoids counter-trend trades

name = "12h_1w_1d_camarilla_pivot_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators for Camarilla calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from prior 1w bar
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    pp_1w = typical_price_1w  # Simplified: PP = (H+L+C)/3
    range_1w = high_1w - low_1w
    r3_1w = pp_1w + (range_1w * 1.1 / 2.0)
    s3_1w = pp_1w - (range_1w * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h timeframe (completed 1w bar only)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price closes above R3 with volume spike and 1d uptrend
            if (prices['close'].iloc[i] > r3_1w_aligned[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price closes below S3 with volume spike and 1d downtrend
            elif (prices['close'].iloc[i] < s3_1w_aligned[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to Camarilla pivot point (PP)
            if position == 1 and prices['close'].iloc[i] < pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals