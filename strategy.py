#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Bands Squeeze Breakout with 1d ADX trend filter and volume confirmation
# Bollinger Squeeze (BB width at low percentile) identifies low volatility periods.
# Breakout from squeeze in direction of 1d ADX trend (>25) with volume confirmation (>1.5x avg).
# Works in both bull and bear markets by aligning with daily trend direction, avoiding false breakouts.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        dx[period+1:] = 100 * np.abs(plus_di[period+1:] - minus_di[period+1:]) / (plus_di[period+1:] + minus_di[period+1:])
        
        adx = np.zeros_like(high)
        adx[2*period+1] = np.mean(dx[period+1:2*period+2])
        for i in range(2*period+2, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Bollinger Bands (20, 2) on 12h data
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * bb_stddev
    lower = sma - bb_std * bb_stddev
    bb_width = (upper - lower) / sma  # Normalized width
    
    # Bollinger Squeeze: BB width at 20th percentile (low volatility)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bb_width <= bb_width_percentile
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):
        # Skip if data not ready
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(bb_stddev[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger Squeeze breakout up + ADX > 25 + volume spike
            if (squeeze[i] and 
                close[i] > upper[i] and 
                adx_14_1d_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger Squeeze breakout down + ADX > 25 + volume spike
            elif (squeeze[i] and 
                  close[i] < lower[i] and 
                  adx_14_1d_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of Bollinger Bands or volatility expands
            if position == 1:
                # Exit long: Price crosses below middle (SMA) or BB width > 50th percentile
                if (close[i] < sma[i] or 
                    bb_width[i] > pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.50).values[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price crosses above middle (SMA) or BB width > 50th percentile
                if (close[i] > sma[i] or 
                    bb_width[i] > pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.50).values[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_BollingerSqueeze_1dADX25_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0