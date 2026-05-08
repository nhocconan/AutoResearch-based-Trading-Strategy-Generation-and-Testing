#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1wTrend_Volume_Squeeze"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = high_1d[0]
    plow[0] = low_1d[0]
    pclose[0] = close_1d[0]
    
    # Camarilla levels
    range_ = phigh - plow
    camarilla_r3 = pclose + (range_ * 1.1 / 2)
    camarilla_s3 = pclose - (range_ * 1.1 / 2)
    camarilla_r4 = pclose + (range_ * 1.1)
    camarilla_s4 = pclose - (range_ * 1.1)
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Weekly trend filter: EMA 34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Bollinger Band width for squeeze detection (20, 2)
    close_series = pd.Series(close)
    bb_ma = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma
    bb_width = np.nan_to_num(bb_width, nan=0.0)
    # Squeeze: BB width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(bb_squeeze[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume, weekly uptrend, and volatility squeeze
            if (close[i] > r3_1d_aligned[i] and 
                vol_ratio[i] > 1.8 and
                close[i] > ema_34_1w_aligned[i] and
                bb_squeeze[i]):
                # Avoid extreme extension beyond R4
                if close[i] <= r4_1d_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: break below S3 with volume, weekly downtrend, and volatility squeeze
            elif (close[i] < s3_1d_aligned[i] and 
                  vol_ratio[i] > 1.8 and
                  close[i] < ema_34_1w_aligned[i] and
                  bb_squeeze[i]):
                # Avoid extreme extension beyond S4
                if close[i] >= s4_1d_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below S3 or weekly trend turns down
            if close[i] < s3_1d_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above R3 or weekly trend turns up
            if close[i] > r3_1d_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals