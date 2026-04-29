#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 24-bar avg
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 24-bar avg
# Exit when price retouches Camarilla pivot point (mean reversion) or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# Camarilla levels from 1d provide institutional reference points; breakout captures momentum.
# Volume spike ensures breakouts have conviction, reducing false signals in choppy markets.
# 1d EMA34 filter ensures alignment with daily trend, improving performance in both bull and bear markets.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first value to NaN since we don't have prior day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    pp = pivot  # Pivot point for exit
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: >2.0x 24-bar average volume (24*12h = 12 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Volume MA(24) needs 24 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_spike[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        pp_val = pp_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if curr_high > r3_val and curr_close > ema_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif curr_low < s3_val and curr_close < ema_val and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches pivot point or breaks below S3
            if curr_close <= pp_val or curr_low < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches pivot point or breaks above R3
            if curr_close >= pp_val or curr_high > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals