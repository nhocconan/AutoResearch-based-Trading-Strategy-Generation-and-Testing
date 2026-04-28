#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme with 1d Volume Spike and 12h EMA50 Trend Filter
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
# Requires 1d volume > 2.0x 20-bar average for confirmation
# Requires 12h EMA50 trend alignment (price > EMA50 for long, price < EMA50 for short)
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 25-40 trades/year via extreme readings + volume + trend confluence
# Works in bull markets via long oversold bounces, bear markets via short overbought rallies

name = "4h_WilliamsR_Extreme_1dVolumeSpike_12hEMA50_TrendFilter_v1"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20) for spike detection
    vol_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > 2.0 * volume_ma_20_1d
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 4h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align HTF indicators to 4h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        vol_spike = volume_spike_1d_aligned[i] > 0.5  # Boolean array converted to float
        ema_50 = ema_50_12h_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND volume spike AND price > 12h EMA50
            if wr < -80 and vol_spike and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND volume spike AND price < 12h EMA50
            elif wr > -20 and vol_spike and price < ema_50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R > -50 (momentum fading) or no volume spike
            if wr > -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R < -50 (momentum fading) or no volume spike
            if wr < -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals