#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band Breakout with Weekly Volume Confirmation and Trend Filter
# Uses Bollinger Bands (20, 2) on daily timeframe for volatility-based breakouts.
# Trades breakouts above upper band or below lower band only when confirmed by:
# 1. Volume > 1.5x 20-day average volume (on daily)
# 2. Weekly trend filter: price above/below weekly 50 EMA (trending market)
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Target: 20-80 total trades over 4 years (5-20/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Bollinger Bands and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2) on daily close
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Calculate 20-day average volume on daily
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly 50 EMA for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe (since we're trading on 1d)
    upper_band_aligned = align_htf_to_ltf(close, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(close, df_1d, lower_band)
    vol_ma_aligned = align_htf_to_ltf(close, df_1d, vol_ma)
    ema_50_aligned = align_htf_to_ltf(close, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    # Start from index 30 to ensure we have enough data for indicators
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_50_aligned[i])):
            continue
        
        # Long entry: price breaks above upper Bollinger Band + volume confirmation + uptrend
        if (close[i] > upper_band_aligned[i] and
            volume[i] > 1.5 * vol_ma_aligned[i] and
            close[i] > ema_50_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower Bollinger Band + volume confirmation + downtrend
        elif (close[i] < lower_band_aligned[i] and
              volume[i] > 1.5 * vol_ma_aligned[i] and
              close[i] < ema_50_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or when price crosses back inside Bollinger Bands
        elif position == 1 and close[i] < sma[-1] if i < len(sma) else False:  # Simplified exit logic
            # Actually, we need to check against the aligned SMA
            sma_aligned = align_htf_to_ltf(close, df_1d, sma)
            if not np.isnan(sma_aligned[i]) and close[i] < sma_aligned[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1 and not np.isnan(sma_aligned[i]) and close[i] > sma_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Breakout_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0