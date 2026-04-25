#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout with 1d Volume Spike and ADX Trend Filter
Hypothesis: Camarilla R3/S3 levels act as significant intraday support/resistance. 
Breakouts above R3 or below S3 with volume confirmation (>1.5x 20-period average) 
and ADX > 25 (trending market) capture strong momentum moves. 
In bull markets: long breakouts above R3. In bear markets: short breakdowns below S3. 
Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 15-30 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    R2 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    PP = (prev_high + prev_low + prev_close) / 3
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    S2 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    R3_1d = align_htf_to_ltf(prices, df_1d, R3)
    S3_1d = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate 1d EMA20 for volume average (call ONCE before loop)
    vol_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    # Calculate 1d ADX for trend filter (call ONCE before loop)
    # ADX calculation: +DM, -DM, TR, then DX, then ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smooth(dx, period)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = 50  # enough for ADX and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or 
            np.isnan(vol_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        r3_level = R3_1d[i]
        s3_level = S3_1d[i]
        vol_avg = vol_20_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > (1.5 * vol_avg)
        
        # Trend filter: ADX > 25 indicates trending market
        strong_trend = adx_val > 25
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 with volume spike and strong trend
            long_entry = (curr_close > r3_level) and volume_spike and strong_trend
            # Short: price breaks below S3 with volume spike and strong trend
            short_entry = (curr_close < s3_level) and volume_spike and strong_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls back below R3 (breakout failed) OR ADX drops below 20 (trend weakening)
            if (curr_close < r3_level) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises back above S3 (breakdown failed) OR ADX drops below 20 (trend weakening)
            if (curr_close > s3_level) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXTrend"
timeframe = "6h"
leverage = 1.0