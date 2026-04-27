#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Works in bull/bear: Buys strength in uptrend, sells weakness in downtrend.
# Volume filter prevents false breakouts. Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h ATR(14) for volatility
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr14_12h)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Volatility filter: ATR below its 50-period median (low volatility regime)
    atr_median = pd.Series(atr14_12h_aligned).rolling(window=50, min_periods=14).median().values
    vol_filter = atr14_12h_aligned < atr_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr14_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i])):
            signals[i] = 0.0
            continue
        
        if i < 2:  # Need at least 2 periods for Camarilla calculation
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels for current 12h bar (using previous bar's OHLC)
        # We use the completed 12h bar at i-1 to calculate levels for bar i
        if i-1 < len(df_12h):
            # Get the index of the 12h bar that corresponds to current time
            # Since we're using aligned data, we can use the current bar's relationship
            # For simplicity, we use the previous completed 12h bar's data
            idx_12h = min(i-1, len(df_12h)-1)
            if idx_12h >= 1:
                prev_high = df_12h['high'].iloc[idx_12h-1]
                prev_low = df_12h['low'].iloc[idx_12h-1]
                prev_close = df_12h['close'].iloc[idx_12h-1]
                
                # Camarilla levels
                range_val = prev_high - prev_low
                r3 = prev_close + (range_val * 1.1 / 4)
                s3 = prev_close - (range_val * 1.1 / 4)
                
                # Long: price breaks above R3 with trend and volume
                if (close[i] > r3 and 
                    close[i] > ema50_1d_aligned[i] and 
                    volume_filter[i] and 
                    vol_filter[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S3 with trend and volume
                elif (close[i] < s3 and 
                      close[i] < ema50_1d_aligned[i] and 
                      volume_filter[i] and 
                      vol_filter[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0