#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume confirmation and 1d EMA50 trend filter
# Bollinger Band squeeze identifies low volatility primed for expansion
# Breakout direction confirmed by 12h volume spike (>2.0x 20-period average)
# Trend filter: price must be above/below 1d EMA50 to align with higher timeframe trend
# Designed for low frequency (50-150 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + volatility breakout logic

name = "6h_BollingerSqueeze_VolumeSpike_12hConfirm_1dEMA50_Trend_v1"
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
    
    # Bollinger Bands (20, 2.0) on 6h
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_bb + (bb_std * bb_std_dev)
    lower_band = sma_bb - (bb_std * bb_std_dev)
    bb_width = (upper_band - lower_band) / sma_bb  # Normalized width
    
    # Bollinger Band squeeze: width below 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # 12h HTF data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (volume_ma_20_12h * 2.0)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(bb_period + 20, 20, 50)  # BB(20) + width MA(20) + EMA(50)
    
    for i in range(start_idx, n):
        if (np.isnan(sma_bb[i]) or np.isnan(bb_width_ma[i]) or 
            np.isnan(volume_spike_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band breakout conditions
        breakout_up = close[i] > upper_band[i] and squeeze[i-1]  # Break above upper band after squeeze
        breakout_down = close[i] < lower_band[i] and squeeze[i-1]  # Break below lower band after squeeze
        
        # Volume confirmation from 12h
        vol_confirm = volume_spike_12h_aligned[i]
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout + volume spike + uptrend
            if breakout_up and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + volume spike + downtrend
            elif breakout_down and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish breakout or squeeze re-formation
            if breakout_down or squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish breakout or squeeze re-formation
            if breakout_up or squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals