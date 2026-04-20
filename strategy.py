#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for pivot levels and volatility
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    
    # Shift to get previous day's values
    prev_high = np.roll(high_daily, 1)
    prev_low = np.roll(low_daily, 1)
    prev_close = np.roll(close_daily, 1)
    
    # Handle first day
    prev_high[0] = high_daily[0]
    prev_low[0] = low_daily[0]
    prev_close[0] = close_daily[0]
    
    # Calculate pivot levels
    range_prev = prev_high - prev_low
    camarilla_r1 = prev_close + range_prev * 1.1 / 12
    camarilla_s1 = prev_close - range_prev * 1.1 / 12
    camarilla_r2 = prev_close + range_prev * 1.1 / 6
    camarilla_s2 = prev_close - range_prev * 1.1 / 6
    camarilla_r3 = prev_close + range_prev * 1.1 / 4
    camarilla_s3 = prev_close - range_prev * 1.1 / 4
    camarilla_r4 = prev_close + range_prev * 1.1 / 2
    camarilla_s4 = prev_close - range_prev * 1.1 / 2
    
    # Align pivot levels to lower timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s2)
    r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # Daily ATR for volatility filter
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily volume average for confirmation
    volume_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_daily_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        atr_daily = atr_daily_aligned[i]
        vol_ma_daily = vol_ma_daily_aligned[i]
        vol_current = volume[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter_ok = atr_daily > 0
        
        # Volume filter: current volume > 1.3x daily average
        vol_ok = vol_current > 1.3 * vol_ma_daily
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if price > r1 and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif price < s1 and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 or volatility drops significantly
            if price < s1 or not vol_filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 or volatility drops significantly
            if price > r1 or not vol_filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_R1S1_Breakout_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0