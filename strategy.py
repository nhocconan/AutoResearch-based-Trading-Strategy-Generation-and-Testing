#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Williams %R extremes + 4h EMA trend filter + volume confirmation.
Long when Williams %R(14) crosses above -80 (oversold bounce) with 4h EMA50 > EMA200 (uptrend) and volume > 1.3x 20-period volume average.
Short when Williams %R(14) crosses below -20 (overbought rejection) with 4h EMA50 < EMA200 (downtrend) and volume > 1.3x 20-period volume average.
Williams %R identifies exhaustion points in ranging markets; 4h EMA filter ensures we only trade with the intermediate trend; volume confirms conviction.
Designed to work in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) with mean-reversion entries.
Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate Williams %R(14) on 1d
    def williams_r(high_vals, low_vals, close_vals, window):
        highest_high = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_vals) / (highest_high - lowest_low)
        return wr
    
    wr_14_1d = williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate 4h EMA50 and EMA200 for trend
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_50_4h = ema(close_4h, 50)
    ema_200_4h = ema(close_4h, 200)
    
    # Calculate 20-period volume average on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (6h)
    wr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_14_1d)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)  # align volume MA to 6h via 4h for stability
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_14_1d_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_aligned[i]
        # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
        uptrend = ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        downtrend = ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        
        # Williams %R signals: cross above -80 (long), cross below -20 (short)
        wr_long_signal = (wr_14_1d_aligned[i] > -80) and (wr_14_1d_aligned[i-1] <= -80) if i > 0 else False
        wr_short_signal = (wr_14_1d_aligned[i] < -20) and (wr_14_1d_aligned[i-1] >= -20) if i > 0 else False
        
        if position == 0:
            # Long: Williams %R crosses above -80 with uptrend and volume
            if wr_long_signal and uptrend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 with downtrend and volume
            elif wr_short_signal and downtrend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) or trend breaks
            if (wr_14_1d_aligned[i] > -20) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) or trend breaks
            if (wr_14_1d_aligned[i] < -80) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dWilliamsR14_4hEMA50_200_Volume_Confirm"
timeframe = "6h"
leverage = 1.0