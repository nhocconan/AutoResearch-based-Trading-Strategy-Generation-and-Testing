#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Spike and 1w EMA200 Trend Filter
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets
# Breakout confirmed by volume spike (>2.0x 20-period average) and 1w EMA200 trend filter
# Designed for low frequency (50-150 trades over 4 years) to minimize fee drag
# Works in bull/bear via volatility contraction/expansion cycle + trend filter

name = "6h_BollingerSqueeze_VolumeSpike_1wEMA200_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w HTF data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Bollinger Bands (20, 2.0) on 1d data
    close_1d = df_1d['close'].values
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2.0 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2.0 * std_20_1d)
    
    # Bollinger Band Width (normalized by middle band)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    
    # Bollinger Band Squeeze condition: BB Width < 20-period average BB Width
    bb_width_ma_20_1d = pd.Series(bb_width_1d).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width_1d < bb_width_ma_20_1d
    
    # Align Bollinger Band levels and squeeze condition to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition.astype(float))
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(200, 20)  # Need 1w EMA200 and Bollinger Bands
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(squeeze_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band breakout conditions
        breakout_up = close[i] > upper_bb_aligned[i-1]  # Break above upper BB
        breakout_down = close[i] < lower_bb_aligned[i-1]  # Break below lower BB
        
        # Trend filter: price above/below 1w EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation and squeeze condition
        vol_spike = volume_spike[i]
        is_squeeze = squeeze_aligned[i-1] > 0.5  # Was in squeeze on previous bar
        
        if position == 0:  # Flat - look for new entries
            # Long: upward breakout above upper BB, volume spike, was in squeeze, uptrend
            if breakout_up and vol_spike and is_squeeze and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout below lower BB, volume spike, was in squeeze, downtrend
            elif breakout_down and vol_spike and is_squeeze and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on trend reversal or price re-enters Bollinger Bands (below upper BB)
            if not uptrend or close[i] < upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on trend reversal or price re-enters Bollinger Bands (above lower BB)
            if not downtrend or close[i] > lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals