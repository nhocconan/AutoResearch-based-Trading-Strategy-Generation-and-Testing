#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_1dVolFilter
Hypothesis: On 1h timeframe, Camarilla R3/S3 breakouts from previous 1h bar with 4h EMA50 trend filter and 1d volume spike (>2.0x 20-bar avg) captures institutional breakouts with controlled trade frequency. The 1h timeframe targets 15-37 trades/year (60-150 over 4 years), using 4h for signal direction and 1h only for entry timing. Volume filter ensures participation, and discrete sizing (0.20) minimizes fee churn. Works in bull markets via long breakouts and bear markets via short breakouts.
"""

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
    
    # Get 4h data for HTF trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1h data for Camarilla levels (primary timeframe)
    # We need to resample 1h data ourselves for Camarilla calculation since we're using 1h as primary
    # But we must do it correctly - we'll create 1h OHLC from the prices dataframe
    # Since prices is already 1h, we can use it directly
    
    # Calculate Camarilla levels from previous 1h bar (R3, S3)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Use previous completed 1h bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Volume average (20-period) for volume spike filter on 1h
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(r3[i]) or 
            np.isnan(s3[i]) or 
            np.isnan(vol_ma_1h[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        r3_val = r3[i]
        s3_val = s3[i]
        vol_ma_1h_val = vol_ma_1h[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 1h volume > 2.0x 20-period average AND 1d volume > 2.0x 20-period average
        volume_spike_1h = vol_val > 2.0 * vol_ma_1h_val
        volume_spike_1d = volume_1d[i // 24] > 2.0 * vol_ma_1d_val if i // 24 < len(volume_1d) else False
        volume_spike = volume_spike_1h and volume_spike_1d
        
        if position == 0:
            # Look for entry signals: Camarilla R3/S3 breakout with trend and volume
            # Long: price breaks above R3 with uptrend (close > EMA50) and volume spike
            long_signal = (high_val > r3_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below S3 with downtrend (close < EMA50) and volume spike
            short_signal = (low_val < s3_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S3 (exit long)
            if close_val < s3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R3 (exit short)
            if close_val > r3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0