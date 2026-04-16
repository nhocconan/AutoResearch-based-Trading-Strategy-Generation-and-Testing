#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation and 1d ADX trend filter.
# Long when price breaks above R4 with volume spike and 1d ADX > 25 (strong trend).
# Short when price breaks below S4 with volume spike and 1d ADX > 25.
# Exit when price retraces to the 1d VWAP (mean reversion to fair value).
# Uses discrete position size 0.25. Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 50-150 total trades over 4 years (12-37/year). Uses 1d for trend/structure, 12h for pivot levels, 6h for execution.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data once before loop for ADX and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX(14) and VWAP ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate ADX components
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilders_smoothing(tr, period_adx)
    plus_dm_smooth = wilders_smoothing(plus_dm, period_adx)
    minus_dm_smooth = wilders_smoothing(minus_dm, period_adx)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, period_adx)
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price * vol_1d)
    vwap_den = np.cumsum(vol_1d)
    vwap = vwap_num / (vwap_den + 1e-10)
    
    # Get 12h data once before loop for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h Indicators: Camarilla pivot levels (based on previous day) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels use previous day's close, high, low
    # For 12h timeframe, we use the prior 12h bar's OHLC
    prev_close = np.concatenate([[close_12h[0]], close_12h[:-1]])
    prev_high = np.concatenate([[high_12h[0]], high_12h[:-1]])
    prev_low = np.concatenate([[low_12h[0]], low_12h[:-1]])
    
    range_ = prev_high - prev_low
    camarilla_mult = 1.1 / 12  # Camarilla multiplier
    
    # R levels (resistance)
    r3 = prev_close + range_ * camarilla_mult * 3
    r4 = prev_close + range_ * camarilla_mult * 4
    # S levels (support)
    s3 = prev_close - range_ * camarilla_mult * 3
    s4 = prev_close - range_ * camarilla_mult * 4
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # === 6h Indicators: Volume median for spike detection ===
    vol_6h = df_6h['volume'].values
    vol_median_20 = pd.Series(vol_6h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    vol_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period_adx*2, 20, 20)  # ADX needs 2*period for smoothing, others need 20
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        adx_val = adx_aligned[i]
        vwap = vwap_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_6h = vol_6h_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price retraces to VWAP (mean reversion)
            if price <= vwap:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price retraces to VWAP (mean reversion)
            if price >= vwap:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 6h volume > 1.5x median volume
            volume_spike = vol_6h > (vol_median * 1.5)
            # Strong trend filter: ADX > 25
            strong_trend = adx_val > 25
            
            # LONG CONDITIONS
            # Price breaks above R4 resistance AND volume spike AND strong trend
            if price > r4 and volume_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below S4 support AND volume spike AND strong trend
            elif price < s4 and volume_spike and strong_trend:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_Camarilla_R4S4_6hVolumeSpike1.5x_1dADX25_v1"
timeframe = "6h"
leverage = 1.0