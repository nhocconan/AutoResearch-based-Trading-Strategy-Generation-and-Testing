#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 4h EMA50 trend filter
# Long when price breaks above R3 AND volume > 2x 20-bar average AND close > EMA50
# Short when price breaks below S3 AND volume > 2x 20-bar average AND close < EMA50
# Exit when price re-enters Camarilla H3/L3 range or volume drops below average
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 20-50 trades/year via tight entry conditions requiring confluence

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_4hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    R2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    PP = (high_1d + low_1d + close_1d) / 3
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    S2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    H3 = R1  # Camarilla H3 equals R1
    L3 = S1  # Camarilla L3 equals S1
    
    # Prepend NaN for first bar (no previous day)
    R3 = np.concatenate([np.array([np.nan]), R3[:-1]])
    S3 = np.concatenate([np.array([np.nan]), S3[:-1]])
    H3 = np.concatenate([np.array([np.nan]), H3[:-1]])
    L3 = np.concatenate([np.array([np.nan]), L3[:-1]])
    
    # Align 1d Camarilla levels to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    
    # 4h EMA(50) for trend filter
    close_s = pd.Series(close)
    ema_50_4h = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_spike[i]
        ema_trend = ema_50_4h[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND volume spike AND close > EMA50
            if price > R3_4h[i] and vol_conf and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND volume spike AND close < EMA50
            elif price < S3_4h[i] and vol_conf and price < ema_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price re-enters H3/L3 range or volume drops
            if price < H3_4h[i] or price > L3_4h[i] or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price re-enters H3/L3 range or volume drops
            if price > L3_4h[i] or price < H3_4h[i] or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals