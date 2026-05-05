#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and 12h EMA50 trend filter
# Long when price breaks above Camarilla R1 AND 1d volume > 2.0x 20-period average AND 12h close > 12h EMA50
# Short when price breaks below Camarilla S1 AND 1d volume > 2.0x 20-period average AND 12h close < 12h EMA50
# Exit when price crosses 12h EMA50 (trend reversal)
# Uses 4h primary timeframe with 1d HTF for volume confirmation and 12h HTF for trend filter
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) based on proven Camarilla breakout performance
# Works in both bull and bear markets by following the 12h trend while using 4h for entry timing and 1d for volume confirmation

name = "4h_Camarilla_R1_S1_Breakout_1dVolumeSpike_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate volume spike on 1d data: volume > 2.0x 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels on 4h data (based on previous bar's range)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    if len(high) >= 2:
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        camarilla_range = prev_high - prev_low
        camarilla_upper = prev_close + (1.1 * camarilla_range / 12)
        camarilla_lower = prev_close - (1.1 * camarilla_range / 12)
    else:
        camarilla_upper = np.full(n, np.nan)
        camarilla_lower = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_upper[i]) or 
            np.isnan(camarilla_lower[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND 1d volume spike AND 12h close > 12h EMA50
            if (close[i] > camarilla_upper[i] and 
                volume_spike_1d_aligned[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND 1d volume spike AND 12h close < 12h EMA50
            elif (close[i] < camarilla_lower[i] and 
                  volume_spike_1d_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals