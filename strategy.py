#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2x 20-bar avg
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2x 20-bar avg
# Exit when price reverts to R4/S4 levels or volume drops below average
# Target: 12-37 trades/year via tight breakout conditions + volume confirmation
# Works in both bull and bear markets by using Camarilla levels (adaptive to volatility) + trend filter

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need sufficient data for EMA34 and Camarilla
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels (R3, R4, S3, S4) on 1d data
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using previous bar's OHLC)
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    r4 = close_1d + range_1d * 1.1 / 2
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Prepend NaN for first bar (no previous day)
    r3 = np.concatenate([[np.nan], r3[:-1]])
    r4 = np.concatenate([[np.nan], r4[:-1]])
    s3 = np.concatenate([[np.nan], s3[:-1]])
    s4 = np.concatenate([[np.nan], s4[:-1]])
    ema_34_1d = np.concatenate([np.full(34, np.nan), ema_34_1d])  # EMA warmup
    
    # Align 1d indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_spike[i]
        price = close[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND trend up AND volume spike
            if price > r3_aligned[i] and price > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND trend down AND volume spike
            elif price < s3_aligned[i] and price < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price reaches R4 or trend changes or no volume
            if price >= r4_aligned[i] or price < ema_trend or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price reaches S4 or trend changes or no volume
            if price <= s4_aligned[i] or price > ema_trend or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals