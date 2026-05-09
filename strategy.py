#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_LiquiditySweep_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Lookback for liquidity sweep detection (last 24 bars = 4 days on 6h)
    lookback = 24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ma20_current[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Check for liquidity sweep (stop hunt) and reversal
            # Look for recent low followed by strong close above (bullish)
            # or recent high followed by strong close below (bearish)
            recent_low = np.min(low[i-lookback:i])
            recent_high = np.max(high[i-lookback:i])
            
            bullish_sweep = (low[i] <= recent_low * 1.001) and (close[i] > recent_low * 1.02)
            bearish_sweep = (high[i] >= recent_high * 0.999) and (close[i] < recent_high * 0.98)
            
            # Long: bullish sweep with volume and above 1d EMA trend
            if bullish_sweep and vol_ok and (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish sweep with volume and below 1d EMA trend
            elif bearish_sweep and vol_ok and (close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to sweep level or trend reversal
            recent_low = np.min(low[i-lookback:i])
            if (close[i] <= recent_low * 1.005) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to sweep level or trend reversal
            recent_high = np.max(high[i-lookback:i])
            if (close[i] >= recent_high * 0.995) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals