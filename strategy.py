#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band Squeeze breakout with 1d EMA34 trend filter and volume spike confirmation
# Bollinger Band squeeze (low volatility) precedes breakouts in both bull and bear markets
# Breakout direction confirmed by 1d EMA34 trend and volume spike
# Works in bull markets via upside breakouts and in bear markets via downside breakouts
# Target: 12-37 trades/year on 12h timeframe to minimize fee drag

name = "12h_BB_Squeeze_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bollinger Bands (20, 2) on 12h
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * bb_std_dev)
    lower_band = sma_20 - (bb_std * bb_std_dev)
    bb_width = (upper_band - lower_band) / sma_20  # Normalized BB width
    
    # Bollinger Band Squeeze: BB width below 20-period average (low volatility)
    bb_width_ma_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma_20
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, bb_period, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(bb_width[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze breakout above upper band, volume spike, uptrend
            if close[i] > upper_band[i] and bb_squeeze[i] and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower band, volume spike, downtrend
            elif close[i] < lower_band[i] and bb_squeeze[i] and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below middle band (SMA20) or trend reversal
            if close[i] < sma_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above middle band (SMA20) or trend reversal
            if close[i] > sma_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals