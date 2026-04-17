#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d timeframe Camarilla pivot levels (R1/S1, R2/S2, R3/S3, R4/S4).
Enter long on breakout above R4 with volume confirmation; enter short on breakdown below S4 with volume confirmation.
Exit when price returns to the 1d VWAP (volume-weighted average price) or opposite pivot level.
Camarilla pivots derived from 1d OHLC provide mathematically precise support/resistance levels that work across market regimes.
Volume confirmation filters false breakouts. Designed to capture strong momentum moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla formulas: 
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.25 * (high - low)
    # R2 = close + 1.166 * (high - low)
    # R1 = close + 0.833 * (high - low)
    # PP = (high + low + close) / 3
    # S1 = close - 0.833 * (high - low)
    # S2 = close - 1.166 * (high - low)
    # S3 = close - 1.25 * (high - low)
    # S4 = close - 1.5 * (high - low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first bar
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    rang = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r4 = prev_close + 1.5 * rang
    camarilla_r3 = prev_close + 1.25 * rang
    camarilla_r2 = prev_close + 1.166 * rang
    camarilla_r1 = prev_close + 0.833 * rang
    camarilla_s1 = prev_close - 0.833 * rang
    camarilla_s2 = prev_close - 1.166 * rang
    camarilla_s3 = prev_close - 1.25 * rang
    camarilla_s4 = prev_close - 1.5 * rang
    
    # Calculate 1d VWAP (typical price * volume cumsum / volume cumsum)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vol_cumsum = np.cumsum(volume_1d)
    tp_vol_cumsum = np.cumsum(typical_price * volume_1d)
    vwap_1d = np.where(vol_cumsum > 0, tp_vol_cumsum / vol_cumsum, typical_price)
    
    # Align all to primary timeframe (6h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 1  # need previous day's data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average (using available history)
        if i >= 20:
            vol_ma_20 = np.mean(volume[max(0, i-19):i+1])
            volume_confirmed = volume[i] > 1.5 * vol_ma_20
        else:
            volume_confirmed = True  # no volume filter early on
        
        if position == 0:
            # Long: price breaks above R4 with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP or falls below R3 (taking partial profit)
            if (close[i] <= vwap_1d_aligned[i] or close[i] < camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to VWAP or rises above S3 (taking partial profit)
            if (close[i] >= vwap_1d_aligned[i] or close[i] > camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dCamarilla_R4S4_Breakout_VWAP_Exit_Volume_Confirm"
timeframe = "6h"
leverage = 1.0