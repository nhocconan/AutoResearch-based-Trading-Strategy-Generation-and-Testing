#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot (Camarilla) Reversal + Volume Confirmation
# Uses weekly Camarilla pivot levels (calculated from prior week's range) to identify
# reversal zones at S3/R3 (strong reversal) and breakout continuation at S4/R4.
# Trades only in direction of weekly trend (price vs weekly open) to avoid counter-trend.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Designed for low frequency: ~15-30 trades/year to minimize fee drag.
name = "6h_WeeklyCamarilla_Reversal_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots (ONCE before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    # Based on prior week's high, low, close
    H = df_w['high'].values
    L = df_w['low'].values
    C = df_w['close'].values
    
    # Camarilla levels
    R4 = C + ((H - L) * 1.5000)
    R3 = C + ((H - L) * 1.2500)
    R2 = C + ((H - L) * 1.1666)
    R1 = C + ((H - L) * 1.0833)
    PP = (H + L + C) / 3.0
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar close)
    R4_w = align_htf_to_ltf(prices, df_w, R4)
    R3_w = align_htf_to_ltf(prices, df_w, R3)
    S3_w = align_htf_to_ltf(prices, df_w, S3)
    S4_w = align_htf_to_ltf(prices, df_w, S4)
    PP_w = align_htf_to_ltf(prices, df_w, PP)
    
    # Weekly trend filter: price vs weekly open (prior week's open)
    O = df_w['open'].values
    weekly_open_aligned = align_htf_to_ltf(prices, df_w, O)
    weekly_uptrend = close > weekly_open_aligned
    weekly_downtrend = close < weekly_open_aligned
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R3_w[i]) or np.isnan(S3_w[i]) or np.isnan(R4_w[i]) or
            np.isnan(S4_w[i]) or np.isnan(PP_w[i]) or np.isnan(weekly_open_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long reversal: price at S3 with rejection AND weekly uptrend AND volume spike
            # Rejection: close > S3 and low touched S3 or below
            if (close[i] > S3_w[i] and low[i] <= S3_w[i] and weekly_uptrend[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short reversal: price at R3 with rejection AND weekly downtrend AND volume spike
            elif (close[i] < R3_w[i] and high[i] >= R3_w[i] and weekly_downtrend[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            # Long breakout: price breaks above R4 with weekly uptrend AND volume spike
            elif (close[i] > R4_w[i] and weekly_uptrend[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S4 with weekly downtrend AND volume spike
            elif (close[i] < S4_w[i] and weekly_downtrend[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot OR trend reverses
            if close[i] < PP_w[i] or not weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot OR trend reverses
            if close[i] > PP_w[i] or not weekly_downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals