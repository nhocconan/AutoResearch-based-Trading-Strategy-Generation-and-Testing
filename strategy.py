#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter
# Camarilla pivot levels provide precise support/resistance based on prior day's range
# Breakout above R3 or below S3 with volume confirmation indicates strong momentum
# Choppiness index > 61.8 identifies ranging markets (mean revert), < 38.2 identifies trending (breakout follow)
# Only take breakouts in trending regimes (CHOP < 38.2) to avoid false breakouts in ranging markets
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1dChop38_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Choppiness Index for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) - Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan, dtype=float)
        if len(x) < period:
            return result
        result[period-1] = np.nansum(x[:period])
        for i in range(period, len(x)):
            result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    
    # Sum of ATR over 14 periods
    sum_tr_14 = np.zeros_like(atr_14)
    for i in range(len(sum_tr_14)):
        if i < 13:
            sum_tr_14[i] = np.nan
        else:
            sum_tr_14[i] = np.nansum(atr_14[i-13:i+1])
    
    # High-Low range over 14 periods
    max_high_14 = np.zeros_like(high_1d)
    min_low_14 = np.zeros_like(low_1d)
    for i in range(len(max_high_14)):
        if i < 13:
            max_high_14[i] = np.nan
            min_low_14[i] = np.nan
        else:
            max_high_14[i] = np.nanmax(high_1d[i-13:i+1])
            min_low_14[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    range_14 = max_high_14 - min_low_14
    chop = np.full_like(close_1d, np.nan)
    mask = (sum_tr_14 > 0) & (range_14 > 0) & (~np.isnan(sum_tr_14)) & (~np.isnan(range_14))
    chop[mask] = 100 * np.log10(sum_tr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels based on prior day's OHLC
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.25 / 2)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.25 / 2)
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*12h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 24)  # warmup for CHOP and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Regime filter: only take breakouts in trending markets (CHOP < 38.2)
        if chop_aligned[i] >= 38.2:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: break above R3 with close > R3
                if curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S3 with close < S3
                elif curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R3 (breakout fails)
            if curr_close < curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above S3 (breakdown fails)
            if curr_close > curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals