#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v2_optimized"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (based on previous day)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r4 = close_1d + range_1d * 1.1 / 2
    r3 = close_1d + range_1d * 1.1 / 4
    r2 = close_1d + range_1d * 1.1 / 6
    r1 = close_1d + range_1d * 1.1 / 12
    # Support levels
    s1 = close_1d - range_1d * 1.1 / 12
    s2 = close_1d - range_1d * 1.1 / 6
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Shift levels to avoid look-ahead (use previous day's levels for current day)
    r4 = np.roll(r4, 1)
    r3 = np.roll(r3, 1)
    r2 = np.roll(r2, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    s2 = np.roll(s2, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    # Set first day to NaN
    r4[0] = r3[0] = r2[0] = r1[0] = s1[0] = s2[0] = s3[0] = s4[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 4-day average volume for confirmation
    vol_avg_4 = pd.Series(volume_1d).rolling(window=4, min_periods=4).mean().values
    vol_avg_4_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_4)
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_avg_4_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current daily volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_4_aligned[i]
        
        price = close[i]
        
        # Breakout conditions with volume confirmation
        # Long when price breaks above R3 with volume
        long_breakout = (price > r3_aligned[i]) and vol_confirm
        # Short when price breaks below S3 with volume
        short_breakout = (price < s3_aligned[i]) and vol_confirm
        
        # Volatility filter: avoid trading in extremely low volatility
        # Only trade when ATR is above 40% of its 20-period average
        atr_ma_20 = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean()
        atr_ma_20_val = atr_ma_20.iloc[i] if hasattr(atr_ma_20, 'iloc') else atr_ma_20[i] if i < len(atr_ma_20) else np.nan
        vol_filter = not np.isnan(atr_ma_20_val) and atr_1d_aligned[i] > (0.4 * atr_ma_20_val)
        
        if long_breakout and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (price < r1_aligned[i] or not vol_filter):
            # Exit long when price returns to R1 or volatility drops
            position = 0
            signals[i] = 0.0
        elif position == -1 and (price > s1_aligned[i] or not vol_filter):
            # Exit short when price returns to S1 or volatility drops
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals