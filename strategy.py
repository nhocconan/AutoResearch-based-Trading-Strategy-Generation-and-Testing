#!/usr/bin/env python3
# 12h_1d_pivot_breakout_volume_v3
# Hypothesis: 12h Camarilla pivot breakouts with volume confirmation and 1d trend filter (close > EMA50 for long, < EMA50 for short).
# Long: price breaks above R4 (1d) with volume > 2.0x 20-period average AND 1d close > EMA50 (bullish trend)
# Short: price breaks below S4 (1d) with volume > 2.0x 20-period average AND 1d close < EMA50 (bearish trend)
# Exit: price returns to 1d VWAP or opposite pivot level (R3/S3) with volume > 1.5x average
# Uses 12h primary timeframe with 1d HTF for pivot levels, VWAP, and EMA50 trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_pivot_breakout_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for Camarilla pivot levels, VWAP, and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    vwap_1d = np.full(len(df_1d), np.nan)
    ema50_1d = np.full(len(df_1d), np.nan)
    
    # Calculate EMA50 for trend filter
    if len(df_1d) >= 50:
        ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(len(df_1d)):
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        if i == 0:
            vwap_1d[i] = typical_price
        else:
            vwap_1d[i] = (vwap_1d[i-1] * np.sum(volume[:i]) + typical_price * volume[i]) / (np.sum(volume[:i]) + volume[i])
        
        if i > 0:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            range_val = prev_high - prev_low
            
            camarilla_r4[i] = prev_close + range_val * 1.1 / 2.0
            camarilla_r3[i] = prev_close + range_val * 1.1 / 4.0
            camarilla_s3[i] = prev_close - range_val * 1.1 / 4.0
            camarilla_s4[i] = prev_close - range_val * 1.1 / 2.0
    
    # Align 1d indicators to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vwap1d = vwap_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        
        if np.isnan(r4) or np.isnan(r3) or np.isnan(s3) or np.isnan(s4) or np.isnan(vwap1d) or np.isnan(ema50):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to 1d VWAP or breaks below R3 with volume
            if price <= vwap1d or (price < r3 and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to 1d VWAP or breaks above S3 with volume
            if price >= vwap1d or (price > s3 and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above R4 with volume AND 1d close > EMA50 (bullish trend)
            if price > r4 and vol_r > 2.0 and close_1d_search(i, close_1d, ema50_1d) > ema50_1d_search(i, close_1d, ema50_1d):
                position = 1
                entry_price = price
                signals[i] = 0.25
            # Short entry: price breaks below S4 with volume AND 1d close < EMA50 (bearish trend)
            elif price < s4 and vol_r > 2.0 and close_1d_search(i, close_1d, ema50_1d) < ema50_1d_search(i, close_1d, ema50_1d):
                position = -1
                entry_price = price
                signals[i] = -0.25
    
    return signals

def close_1d_search(i, close_1d, ema50_1d):
    """Helper to get the corresponding 1d close for alignment check"""
    # This is a placeholder - in reality we'd use the aligned arrays
    # For now, we'll use a simplified approach
    return close_1d[-1] if len(close_1d) > 0 else 0

# Actually, let's fix the trend check properly by using aligned close and ema50
# We need to pass the aligned 1d close and ema50 to the loop
# Let me rewrite this properly

#!/usr/bin/env python3
# 12h_1d_pivot_breakout_volume_v3
# Hypothesis: 12h Camarilla pivot breakouts with volume confirmation and 1d trend filter (close > EMA50 for long, < EMA50 for short).
# Long: price breaks above R4 (1d) with volume > 2.0x 20-period average AND 1d close > EMA50 (bullish trend)
# Short: price breaks below S4 (1d) with volume > 2.0x 20-period average AND 1d close < EMA50 (bearish trend)
# Exit: price returns to 1d VWAP or opposite pivot level (R3/S3) with volume > 1.5x average
# Uses 12h primary timeframe with 1d HTF for pivot levels, VWAP, and EMA50 trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_pivot_breakout_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for Camarilla pivot levels, VWAP, and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    vwap_1d = np.full(len(df_1d), np.nan)
    ema50_1d = np.full(len(df_1d), np.nan)
    
    # Calculate EMA50 for trend filter
    if len(df_1d) >= 50:
        ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(len(df_1d)):
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        if i == 0:
            vwap_1d[i] = typical_price
        else:
            vwap_1d[i] = (vwap_1d[i-1] * np.sum(volume[:i]) + typical_price * volume[i]) / (np.sum(volume[:i]) + volume[i])
        
        if i > 0:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            range_val = prev_high - prev_low
            
            camarilla_r4[i] = prev_close + range_val * 1.1 / 2.0
            camarilla_r3[i] = prev_close + range_val * 1.1 / 4.0
            camarilla_s3[i] = prev_close - range_val * 1.1 / 4.0
            camarilla_s4[i] = prev_close - range_val * 1.1 / 2.0
    
    # Align 1d indicators to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)  # For trend check
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vwap1d = vwap_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        close1d = close_1d_aligned[i]
        
        if np.isnan(r4) or np.isnan(r3) or np.isnan(s3) or np.isnan(s4) or np.isnan(vwap1d) or np.isnan(ema50) or np.isnan(close1d):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to 1d VWAP or breaks below R3 with volume
            if price <= vwap1d or (price < r3 and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to 1d VWAP or breaks above S3 with volume
            if price >= vwap1d or (price > s3 and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above R4 with volume AND 1d close > EMA50 (bullish trend)
            if price > r4 and vol_r > 2.0 and close1d > ema50:
                position = 1
                entry_price = price
                signals[i] = 0.25
            # Short entry: price breaks below S4 with volume AND 1d close < EMA50 (bearish trend)
            elif price < s4 and vol_r > 2.0 and close1d < ema50:
                position = -1
                entry_price = price
                signals[i] = -0.25
    
    return signals