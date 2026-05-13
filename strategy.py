#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 2.0x average
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 2.0x average
# Exit when price crosses Camarilla H4/L4 (mean reversion) OR trend reversal (price crosses 1d EMA34)
# Uses 12h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with daily trend filter for BTC/ETH resilience.
# Daily EMA34 provides strong trend filter reducing whipsaw; volume spike confirms breakout authenticity.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 12h data for Camarilla calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels on 12h data (using previous bar's OHLC)
    if len(high_12h) >= 1:
        # Camarilla pivot calculation based on previous period
        H = high_12h
        L = low_12h
        C = close_12h
        
        # Previous bar values (shifted by 1)
        H_prev = np.roll(H, 1)
        L_prev = np.roll(L, 1)
        C_prev = np.roll(C, 1)
        H_prev[0] = np.nan
        L_prev[0] = np.nan
        C_prev[0] = np.nan
        
        P = (H_prev + L_prev + C_prev) / 3.0
        R4 = C_prev + ((H_prev - L_prev) * 1.1 / 2.0)
        R3 = C_prev + ((H_prev - L_prev) * 1.1 / 4.0)
        R2 = C_prev + ((H_prev - L_prev) * 1.1 / 6.0)
        R1 = C_prev + ((H_prev - L_prev) * 1.1 / 12.0)
        S1 = C_prev - ((H_prev - L_prev) * 1.1 / 12.0)
        S2 = C_prev - ((H_prev - L_prev) * 1.1 / 6.0)
        S3 = C_prev - ((H_prev - L_prev) * 1.1 / 4.0)
        S4 = C_prev - ((H_prev - L_prev) * 1.1 / 2.0)
        
        # Align levels to current bar (breakout conditions use current bar vs previous levels)
        R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
        S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
        H4_aligned = align_htf_to_ltf(prices, df_12h, R1)  # H4 = R1
        L4_aligned = align_htf_to_ltf(prices, df_12h, S1)  # L4 = S1
    else:
        R3_aligned = np.full(n, np.nan)
        S3_aligned = np.full(n, np.nan)
        H4_aligned = np.full(n, np.nan)
        L4_aligned = np.full(n, np.nan)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 12h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > R3 AND close > 1d EMA34 AND volume spike
            if close[i] > R3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < S3 AND close < 1d EMA34 AND volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < H4 (mean reversion) OR trend reversal (close < 1d EMA34)
            if close[i] < H4_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > L4 (mean reversion) OR trend reversal (close > 1d EMA34)
            if close[i] > L4_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals