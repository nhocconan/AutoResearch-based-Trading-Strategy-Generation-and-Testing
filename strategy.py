#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme reversal with 12h EMA200 trend filter and volume spike
    # Williams %R < -80 = oversold (long), > -20 = overbought (short) in 12h timeframe
    # EMA200 on 12h defines major trend: only long in uptrend, short in downtrend
    # Volume spike (>2.0x 20-period average) confirms reversal strength
    # Designed for low frequency (target: 15-35 trades/year) to minimize fee drag
    # Works in bull/bear markets by following major trend while catching extremes
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R and EMA200
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Williams %R (14-period)
    williams_r = np.full(len(df_12h), np.nan)
    highest_high = np.full(len(df_12h), np.nan)
    lowest_low = np.full(len(df_12h), np.nan)
    
    for i in range(13, len(df_12h)):
        highest_high[i] = np.max(high_12h[i-13:i+1])
        lowest_low[i] = np.min(low_12h[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_12h[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate 12h EMA200 for trend filter
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period average (6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend
        bullish_trend = close[i] > ema200_12h_aligned[i]
        bearish_trend = close[i] < ema200_12h_aligned[i]
        
        # Entry logic: Williams %R extreme with trend filter and volume spike
        long_entry = False
        short_entry = False
        
        # Long: oversold (%R < -80) in bullish trend with volume spike
        if bullish_trend:
            long_entry = (williams_r_aligned[i] < -80) and volume_spike[i]
        # Short: overbought (%R > -20) in bearish trend with volume spike
        elif bearish_trend:
            short_entry = (williams_r_aligned[i] > -20) and volume_spike[i]
        
        # Exit logic: opposite extreme or trend reversal
        long_exit = bearish_trend and (williams_r_aligned[i] > -20)
        short_exit = bullish_trend and (williams_r_aligned[i] < -80)
        
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

name = "6h_12h_williams_r_extreme_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0