#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla R3/S3 breakout with 1h volume confirmation and 6h EMA trend filter
# Long when price breaks above 1d Camarilla R3 level AND 1h volume > 1.5 * avg_volume(20) AND price > 6h EMA(50)
# Short when price breaks below 1d Camarilla S3 level AND 1h volume > 1.5 * avg_volume(20) AND price < 6h EMA(50)
# Exit when price crosses 1d Camarilla pivot level (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1d Camarilla provides daily structure with proven R3/S3 breakout edge
# Volume confirmation on 1h timeframe filters noise while maintaining trade frequency
# 6h EMA(50) ensures we trade with the intermediate trend (works in bull/bear regimes)

name = "6h_1dCamarillaR3S3_1hVolume_EMA50_v1"
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
    
    # Get 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need sufficient data for pivots
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels based on previous 1d bar
    # Camarilla: R3 = Close + 1.125 * (High - Low), S3 = Close - 1.125 * (High - Low)
    camarilla_r3_1d = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_s3_1d = close_1d - 1.125 * (high_1d - low_1d)
    camarilla_pivot_1d = (high_1d + low_1d + close_1d) / 3.0  # Standard pivot for exit
    
    # Get 1h data ONCE before loop for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:  # Need sufficient data for volume average
        return np.zeros(n)
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_spike_1h = volume_1h > (1.5 * avg_volume_20_1h)
    
    # Calculate 6h EMA(50) for trend filter
    ema_50_6h = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    price_above_ema = close > ema_50_6h
    price_below_ema = close < ema_50_6h
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Align 1h volume spike to 6h timeframe (wait for completed 1h bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1h, volume_spike_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(ema_50_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with volume spike and price > 6h EMA(50)
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                volume_spike_aligned[i] and price_above_ema[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with volume spike and price < 6h EMA(50)
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  volume_spike_aligned[i] and price_below_ema[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals