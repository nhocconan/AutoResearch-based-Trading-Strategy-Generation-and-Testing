#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter.
Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25.
Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25.
Exit when price touches opposite Camarilla level (R2/S2 for long/short) or ADX < 20.
Uses 1d HTF for volume and ADX to avoid false breakouts in low-volume/choppy markets.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for Camarilla calculation (typical close)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    ema_34_1d = pd.Series(typical_price_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nansum(tr[1:period+1]) if np.nansum(tr[1:period+1]) > 0 else 1
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            plus_di[i] = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values[i] / atr[i]) if atr[i] > 0 else 0
            minus_di[i] = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values[i] / atr[i]) if atr[i] > 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100) if (plus_di[i] + minus_di[i]) > 0 else 0
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nanmean(dx[period:2*period]) if np.nanmean(dx[period:2*period]) > 0 else 0
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # 1d volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to LTF
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6h Camarilla levels (based on previous 1d typical price)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # We'll use daily typical price as proxy for close in formula
    lookback_1d = 1  # previous day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)
    camarilla_s2 = np.full(n, np.nan)
    
    for i in range(n):
        # Need previous day's data
        if i < 1:
            continue
        # Approximate: use current 6h bar's high/low with 1d typical price
        # Better: use previous completed 1d bar's typical price and range
        # Simplified: use rolling 1d typical price and range aligned
        pass  # Will calculate properly below
    
    # Proper Camarilla calculation using 1d OHLC
    # For each 6h bar, use the most recent completed 1d bar's data
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_r3_1d = typical_1d + 1.125 * range_1d
    camarilla_s3_1d = typical_1d - 1.125 * range_1d
    camarilla_r2_1d = typical_1d + 0.75 * range_1d
    camarilla_s2_1d = typical_1d - 0.75 * range_1d
    
    # Align Camarilla levels
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Break above R3 AND ADX > 25 AND volume spike
            if price > r3 and adx_val > 25 and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 AND ADX > 25 AND volume spike
            elif price < s3 and adx_val > 25 and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches R2 OR ADX < 20
                if price > r2 or adx_val < 20:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches S2 OR ADX < 20
                if price < s2 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0