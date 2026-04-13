#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels with volume confirmation and 1d trend filter.
# Camarilla levels provide precise support/resistance levels derived from prior day's range.
# Breakout above resistance or breakdown below support with volume confirmation indicates momentum.
# 1d EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR (14-period) for pivot levels and volatility filter
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate daily EMA (21-period) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    ema_1d[0] = close_1d[0]
    alpha = 2 / (21 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(atr[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        # Need to get previous day's data - we'll use the 1d data for this
        # Find the index of the previous completed day in 1d data
        current_time = prices.iloc[i]['open_time']
        # Find previous day's data in 1d dataframe
        prev_day_mask = df_1d['open_time'] < current_time
        if not prev_day_mask.any():
            signals[i] = 0.0
            continue
        
        prev_day_idx = prev_day_mask.sum() - 1  # index of previous day
        if prev_day_idx < 0:
            signals[i] = 0.0
            continue
            
        prev_high = df_1d.iloc[prev_day_idx]['high']
        prev_low = df_1d.iloc[prev_day_idx]['low']
        prev_close = df_1d.iloc[prev_day_idx]['close']
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla resistance levels
        r4 = prev_close + range_val * 1.1 / 2
        r3 = prev_close + range_val * 1.1 / 4
        # Camarilla support levels
        s3 = prev_close - range_val * 1.1 / 4
        s4 = prev_close - range_val * 1.1 / 2
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        atr_val = atr[i]
        daily_ema = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # ATR filter: ATR > 0 (always true, but keeps structure)
        atr_filter = atr_val > 0
        
        if position == 0:
            # Long: price breaks above R3 + volume + price above daily EMA
            if (price > r3 and 
                volume_confirm and 
                atr_filter and
                price > daily_ema):
                position = 1
                signals[i] = position_size
            # Short: price breaks below S3 + volume + price below daily EMA
            elif (price < s3 and 
                  volume_confirm and 
                  atr_filter and
                  price < daily_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S3 OR volume drops
            if (price < s3 or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R3 OR volume drops
            if (price > r3 or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0