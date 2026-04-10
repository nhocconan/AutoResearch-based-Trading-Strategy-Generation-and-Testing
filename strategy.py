#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) AND 1d close > 1d EMA(200) AND 4h volume > 1.2x 20-period average
# - Short when Williams %R(14) crosses below -20 (overbought) AND 1d close < 1d EMA(200) AND 4h volume > 1.2x 20-period average
# - Exit when Williams %R crosses -50 (mean reversion midpoint)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R identifies extreme momentum exhaustion points
# - Daily EMA(200) filter ensures we trade with the long-term trend
# - Volume confirmation reduces false signals
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_williamsr_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Williams %R (14-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    highest_high = rolling_max(high, 14)
    lowest_low = rolling_min(low, 14)
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Avoid division by zero
    
    # Pre-compute 4h volume moving average (20-period)
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_4h = rolling_mean(volume, 20)
    
    # Pre-compute 1d EMA(200)
    close_1d = df_1d['close'].values
    ema_200_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 200:
        # Seed with SMA
        ema_200_1d[199] = np.mean(close_1d[:200])
        # EMA calculation
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 + ema_200_1d[i-1] * 198) / 200
    
    # Align HTF indicators to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma_4h[i]) or np.isnan(ema_200_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 1.2 * vol_ma_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R crosses above -80 AND volume spike AND 1d uptrend
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and vol_spike and 
                close > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R crosses below -20 AND volume spike AND 1d downtrend
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and vol_spike and 
                  close < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses -50 (mean reversion midpoint)
            exit_long = (position == 1 and williams_r[i] < -50 and williams_r[i-1] >= -50)
            exit_short = (position == -1 and williams_r[i] > -50 and williams_r[i-1] <= -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result

def rolling_max(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.max(arr[i - window + 1:i + 1])
    return result

def rolling_min(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result