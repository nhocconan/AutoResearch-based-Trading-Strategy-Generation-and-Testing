# 4h_1d_Pivot_R1S1_Breakout_Volume_Regime_v1
# 4h Camarilla pivot breakout with 1d volume confirmation and 1d RSI regime filter
# Long when price breaks above R1 with volume spike and 1d RSI < 60
# Short when price breaks below S1 with volume spike and 1d RSI > 40
# Exit when price returns to pivot or RSI reverses
# Works in bull/bear via RSI regime filter (avoids overbought shorts in bull, oversold longs in bear)
# Target: 25-50 trades/year (100-200 total) to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_R1S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # R1 = C + (H-L) * 1.1 / 12
    r1 = close_1d + range_1d * 1.1 / 12.0
    # S1 = C - (H-L) * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12.0
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d RSI for regime filter
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.rolling(window=14, min_periods=14).mean()
    avg_loss_1d = loss_1d.rolling(window=14, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Volume spike filter: current 4h volume > 1.5x 20-period average of 4h volume
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 20-period average
        volume_filter = vol_ma_4h[i] > 0 and volume[i] > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Look for long entry: price breaks above R1 + volume spike + 1d RSI < 60 (not overbought)
            if (close[i] > r1_aligned[i] and 
                volume_filter and 
                rsi_1d_aligned[i] < 60):
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below S1 + volume spike + 1d RSI > 40 (not oversold)
            elif (close[i] < s1_aligned[i] and 
                  volume_filter and 
                  rsi_1d_aligned[i] > 40):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot or 1d RSI > 70 (overbought)
            if (close[i] <= pivot_aligned[i] or 
                rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot or 1d RSI < 30 (oversold)
            if (close[i] >= pivot_aligned[i] or 
                rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals