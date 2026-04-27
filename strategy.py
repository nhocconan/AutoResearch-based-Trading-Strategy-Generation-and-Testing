#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Works in bull/bear: Buys breakouts in uptrend, sells breakdowns in downtrend.
# Volume filter prevents false breakouts. Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h ATR(14) for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Volatility filter: ATR below its 50-period median (low volatility regime)
    atr_median = pd.Series(atr14_4h_aligned).rolling(window=50, min_periods=14).median().values
    vol_filter = atr14_4h_aligned < atr_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(atr14_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i])):
            signals[i] = 0.0
            continue
        
        if i < 2:  # Need at least 2 periods for Donchian calculation
            signals[i] = 0.0
            continue
            
        # Calculate Donchian levels for current 4h bar (using previous bar's OHLC)
        # We use the completed 4h bar at i-1 to calculate levels for bar i
        if i-1 < len(df_4h):
            # Get the index of the 4h bar that corresponds to current time
            # Since we're using aligned data, we can use the current bar's relationship
            # For simplicity, we use the previous completed 4h bar's data
            idx_4h = min(i-1, len(df_4h)-1)
            if idx_4h >= 1:
                prev_high = df_4h['high'].iloc[idx_4h-1]
                prev_low = df_4h['low'].iloc[idx_4h-1]
                
                # Donchian levels
                upper = prev_high
                lower = prev_low
                
                # Long: price breaks above upper with trend and volume
                if (close[i] > upper and 
                    close[i] > ema50_12h_aligned[i] and 
                    volume_filter[i] and 
                    vol_filter[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower with trend and volume
                elif (close[i] < lower and 
                      close[i] < ema50_12h_aligned[i] and 
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

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0