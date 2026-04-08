#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_reversal_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend and CCI calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate CCI(20) on 12h data
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    sma_tp_12h = pd.Series(typical_price_12h).rolling(window=20, min_periods=20).mean().values
    mad_tp_12h = pd.Series(typical_price_12h).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad_tp_12h = np.where(mad_tp_12h == 0, 1e-10, mad_tp_12h)
    cci_12h = (typical_price_12h - sma_tp_12h) / (0.015 * mad_tp_12h)
    
    # 12h trend: 34-period EMA
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 6h timeframe
    cci_12h_aligned = align_htf_to_ltf(prices, df_12h, cci_12h)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume filter on 6h: volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter_6h = volume > (vol_ma_6h * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(cci_12h_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_filter_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below 0 or trend fails
            if cci_12h_aligned[i] < 0 or close[i] < ema_34_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above 0 or trend fails
            if cci_12h_aligned[i] > 0 or close[i] > ema_34_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Look for CCI extremes with trend alignment
            cci = cci_12h_aligned[i]
            
            # Long: CCI < -100 (oversold) + bullish trend + volume
            if (cci < -100 and 
                close[i] > ema_34_12h_aligned[i] and 
                vol_filter_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: CCI > 100 (overbought) + bearish trend + volume
            elif (cci > 100 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  vol_filter_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals