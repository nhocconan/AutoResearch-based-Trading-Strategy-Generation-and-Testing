#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels calculated from 1d data (R3/S3 for mean reversion, R4/S4 for breakout)
# Only takes breakout trades in direction of 1d EMA34 trend with volume confirmation (>1.5x average)
# Designed for lower trade frequency (target: 12-37 trades/year) to minimize fee drag on 12h timeframe
# Works in both bull/bear markets by combining pivot structure with trend filter and volume confirmation

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC (use previous day's data to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First element will be rolled from end, but min_periods in alignment handles this
    
    # Calculate Camarilla levels: R3, S3, R4, S4
    camarilla_r4 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 2)
    camarilla_r3 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 4)
    camarilla_s3 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 4)
    camarilla_s4 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 2)
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d candles only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe (completed 1d candles only)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume (less strict than 2.0x for lower TF)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # Need sufficient history for volume MA and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above R4 AND 1d EMA34 uptrend AND volume spike
            if price > r4 and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakout: price breaks below S4 AND 1d EMA34 downtrend AND volume spike
            elif price < s4 and price < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on mean reversion to R3 or S3
            # Exit when price returns to R3 (mean reversion) or breaks below S4 (failed breakout)
            if price < r3 or price < s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on mean reversion to S3 or R3
            # Exit when price returns to S3 (mean reversion) or breaks above R4 (failed breakout)
            if price > s3 or price > r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals