#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme with 1d EMA200 trend filter and volume spike
    # Williams %R < -80 = oversold (long), > -20 = overbought (short) on 1d
    # EMA200 determines trend direction: only trade with trend in bull/bear markets
    # Volume confirmation (>2.0x 6h 20-period average) filters false extremes
    # Designed for low frequency: Williams %R extremes are rare (target: 10-25/year)
    # Works in both bull (buy oversold) and bear (sell overbought) regimes
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    williams_r = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high == lowest_low:
            williams_r[i] = -50.0  # avoid division by zero
        else:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100.0
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average (6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Williams %R extreme with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long: Williams %R < -80 (oversold) in bullish trend with volume spike
        if bullish_trend:
            long_entry = (williams_r_aligned[i] < -80.0) and volume_spike[i]
        # Short: Williams %R > -20 (overbought) in bearish trend with volume spike
        elif bearish_trend:
            short_entry = (williams_r_aligned[i] > -20.0) and volume_spike[i]
        
        # Exit logic: Williams %R returns to neutral zone (-50) or trend reversal
        long_exit = bearish_trend and (williams_r_aligned[i] > -50.0)
        short_exit = bullish_trend and (williams_r_aligned[i] < -50.0)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williams_r_extreme_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0