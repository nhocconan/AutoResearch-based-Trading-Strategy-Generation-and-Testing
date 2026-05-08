#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R3/S3) breakout with 1d trend filter (EMA34) and volume confirmation (volume > 1.3x daily VWAP)
# Long when price breaks above R3 + price > daily EMA34 + volume > 1.3x daily VWAP
# Short when price breaks below S3 + price < daily EMA34 + volume > 1.3x daily VWAP
# Exit when price returns to pivot point (PP)
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeVWAP"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation, trend filter and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily VWAP for volume filter (approximated as typical price * volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap = vwap.values
    
    # Align daily indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Calculate daily Camarilla pivot levels (based on previous day's OHLC)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # PP = (High + Low + Close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3 = close_1d + 1.1 * (high_1d - low_1d)
    s3 = close_1d - 1.1 * (high_1d - low_1d)
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vwap_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.3x daily VWAP (as proxy for institutional interest)
        # Find the most recent completed daily bar
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed daily bar
        
        if idx_1d < 0:
            vol_filter = False
        else:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            # Compare to 20-period average volume for normalization
            vol_ma_20 = df_1d['volume'].rolling(window=20, min_periods=20).mean().iloc[idx_1d]
            vol_filter = vol_1d_current > 1.3 * vol_ma_20 if not pd.isna(vol_ma_20) else False
        
        if position == 0:
            # Look for entry: Camarilla breakout + trend + volume
            long_condition = close[i] > r3_aligned[i] and close[i] > ema_34_aligned[i] and vol_filter
            short_condition = close[i] < s3_aligned[i] and close[i] < ema_34_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals