#!/usr/bin/env python3
# 4h_VolumeSpike_Reversal_1dTrend
# Hypothesis: Combines volume spikes on 4h with 1-day trend reversal signals to capture mean reversion in both bull and bear markets.
# Long when: volume spike (>2x 20-period avg) + price closes below 1-day Bollinger Lower Band (20,2) + 1-day RSI < 30.
# Short when: volume spike + price closes above 1-day Bollinger Upper Band + 1-day RSI > 70.
# Exit when price returns to 1-day Bollinger Middle Band (20-day SMA) or RSI crosses 50.
# Uses volume spike to filter low-activity periods and Bollinger Bands + RSI for overextended conditions.
# Targets 20-40 trades per year on 4h timeframe with position size 0.25.

name = "4h_VolumeSpike_Reversal_1dTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20,2)
    close_1d = pd.Series(df_1d['close'])
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d.rolling(window=20, min_periods=20).std()
    upper_band = (sma_20 + 2 * std_20).values
    lower_band = (sma_20 - 2 * std_20).values
    middle_band = sma_20.values  # 20-day SMA
    
    # Calculate 1-day RSI (14)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).values
    
    # Align 1-day indicators to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_1d, middle_band)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma.values * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup for Bollinger Bands and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(middle_band_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long entry: volume spike + price below lower band + RSI oversold
            if (vol_spike and 
                close[i] < lower_band_aligned[i] and 
                rsi_aligned[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short entry: volume spike + price above upper band + RSI overbought
            elif (vol_spike and 
                  close[i] > upper_band_aligned[i] and 
                  rsi_aligned[i] > 70):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle band or RSI crosses above 50
            if (close[i] >= middle_band_aligned[i] or 
                rsi_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle band or RSI crosses below 50
            if (close[i] <= middle_band_aligned[i] or 
                rsi_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals