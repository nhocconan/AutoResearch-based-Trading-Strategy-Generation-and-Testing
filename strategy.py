#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot (R3/S3) breakout with 1d EMA(34) trend filter and volume confirmation
# Weekly Camarilla levels provide strong support/resistance from smart money accumulation/distribution zones.
# Breakouts at R4/S4 with volume spike indicate institutional participation. 1d EMA(34) ensures alignment with daily trend.
# Designed for low trade frequency (~12-37/year on 6h) to minimize fee drag and work in both bull/bear markets via trend filter.

name = "6h_WeeklyCamarilla_R4S4_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = Pivot + Range * 1.5
    # S4 = Pivot - Range * 1.5
    # R3 = Pivot + Range * 1.25
    # S3 = Pivot - Range * 1.25
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    camarilla_r4 = weekly_pivot + weekly_range * 1.5
    camarilla_s4 = weekly_pivot - weekly_range * 1.5
    camarilla_r3 = weekly_pivot + weekly_range * 1.25
    camarilla_s3 = weekly_pivot - weekly_range * 1.25
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for weekly data alignment
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average
        vol_ma_50 = np.mean(volume[max(0, i-50):i]) if i >= 50 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above weekly R4 with 1d uptrend
                if curr_close > curr_r4 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below weekly S4 with 1d downtrend
                elif curr_close < curr_s4 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: price breaks below weekly S3 (reversion to mean)
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches weekly R3 (take partial profit)
            elif curr_close >= curr_r3:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price breaks above weekly R3 (reversion to mean)
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches weekly S3 (take partial profit)
            elif curr_close <= curr_s3:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals