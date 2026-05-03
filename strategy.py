#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# In bull/bear markets, price often respects Camarilla pivot levels. We go long when
# price breaks above R3 with volume spike and 1w uptrend (price > 1w EMA50), short when
# price breaks below S3 with volume spike and 1w downtrend. Uses discrete position sizing
# to minimize fee churn. Designed for low trade frequency (target: 12-37/year) to avoid
# fee drag while capturing strong directional moves.

name = "12h_Camarilla_R3S3_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data for Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        ema_trend = ema_50_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(r3_level) or np.isnan(s3_level):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime-based entry conditions
        # Long: price breaks above R3 with volume spike and 1w uptrend (price > EMA50)
        long_entry = (close[i] > r3_level) and vol_spike and (close[i] > ema_trend)
        # Short: price breaks below S3 with volume spike and 1w downtrend (price < EMA50)
        short_entry = (close[i] < s3_level) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below R3 (breakout failure) or trend reversal
            if close[i] < r3_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above S3 (breakout failure) or trend reversal
            if close[i] > s3_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals