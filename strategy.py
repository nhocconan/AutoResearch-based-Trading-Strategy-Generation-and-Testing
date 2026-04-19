#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d volume confirmation and 1w trend filter
# - Bollinger Band width < 20th percentile indicates squeeze (low volatility)
# - Breakout: price crosses above upper BB (long) or below lower BB (short)
# - Volume filter: 12h volume > 1.5x 20-period 1d average volume (scaled)
# - Trend filter: 1w EMA(50) - only long when price > EMA50, short when price < EMA50
# - Designed for low-frequency, high-conviction trades in both bull and bear markets
# - Target: 15-30 trades/year to minimize fee drag

name = "12h_BBSqueeze_Breakout_1dVolume_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 12h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band width percentile (lookback 50 periods)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Squeeze condition: BB width < 20th percentile (low volatility)
        squeeze_condition = bb_width_percentile[i] < 0.20
        
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d has 2x 12h bars, so divide by 2
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 2.0)
        
        if position == 0:
            # Look for long entry: squeeze + breakout above upper BB + uptrend + volume
            if (squeeze_condition and 
                close[i] > upper_bb[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: squeeze + breakout below lower BB + downtrend + volume
            elif (squeeze_condition and 
                  close[i] < lower_bb[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to middle BB or trend reversal
            middle_bb = sma_20.iloc[i]
            if close[i] < middle_bb or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to middle BB or trend reversal
            middle_bb = sma_20.iloc[i]
            if close[i] > middle_bb or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals