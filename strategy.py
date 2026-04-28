#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme + 1d Volume Spike + 1h Trend Filter
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
# Requires 1d volume > 2.0x 20-bar average for confirmation
# Requires 1h close > EMA50 for long, < EMA50 for short (trend alignment)
# Uses discrete position sizing (0.25) to minimize fee churn
# Designed to work in both bull and bear markets by requiring volume spike and trend filter
# Target: 20-50 trades/year via strict entry conditions

name = "4h_WilliamsR_Extreme_1dVolumeSpike_1hTrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume moving average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 60:
        return np.zeros(n)
    
    # Calculate 1h EMA(50) for trend
    close_1h = df_1h['close'].values
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(ema_50_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        vol_spike = volume_spike_1d_aligned[i]
        ema_50 = ema_50_1h_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND volume spike AND price > EMA50 (uptrend)
            if wr < -80 and vol_spike and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND volume spike AND price < EMA50 (downtrend)
            elif wr > -20 and vol_spike and price < ema_50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R > -50 (recovery) or no volume spike or trend change
            if wr > -50 or not vol_spike or price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R < -50 (recovery) or no volume spike or trend change
            if wr < -50 or not vol_spike or price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals