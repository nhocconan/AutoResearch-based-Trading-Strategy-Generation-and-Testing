#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h trend filter and volume confirmation.
# Long when: BB width at 20-period low (squeeze), price breaks above upper BB, 12h EMA50 uptrend, volume > 1.5x 20-bar average.
# Short when: BB width at 20-period low (squeeze), price breaks below lower BB, 12h EMA50 downtrend, volume > 1.5x 20-bar average.
# Exit when price returns to middle BB (20-period SMA).
# Bollinger squeeze identifies low volatility precede breakouts. 12h EMA50 filters for intermediate trend alignment.
# Volume confirmation ensures breakout validity. Target: 80-180 total trades over 4 years (20-45/year) for 6h timeframe.

name = "6h_BollingerSqueeze_Breakout_12hEMA50_Trend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2.0 * dev)
    lower_band = basis - (2.0 * dev)
    
    # Bollinger Band Width (normalized by basis) for squeeze detection
    bb_width = (upper_band - lower_band) / basis
    bb_width_ma_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma_20  # BB width below its 20-period average = squeeze
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for BB and 12h EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(squeeze_condition[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_basis = basis[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_squeeze = squeeze_condition[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Squeeze, break above upper BB, 12h uptrend, volume confirmation
            if (curr_squeeze and 
                curr_close > curr_upper and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Squeeze, break below lower BB, 12h downtrend, volume confirmation
            elif (curr_squeeze and 
                  curr_close < curr_lower and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to middle BB (20-period SMA)
            if curr_close <= curr_basis:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to middle BB (20-period SMA)
            if curr_close >= curr_basis:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals