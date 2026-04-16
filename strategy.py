#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25 (trending).
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25 (trending).
# Uses discrete position size 0.30. Camarilla levels provide institutional support/resistance,
# volume spike confirms participation, ADX ensures we trade only in trending markets.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla R3, S3 ===
    # Camarilla levels based on previous bar's range
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # We use the previous bar's high/low to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar: use current
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # === 12h Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get 1d data once before loop for trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX for trend filter ===
    # ADX calculation: +DI, -DI, DX, then ADX
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * EWMA(+DM) / ATR
    # -DI = 100 * EWMA(-DM) / ATR
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWMA(DX)
    
    # Calculate +DM and -DM
    high_diff = high_1d - np.roll(high_1d, 1)
    low_diff = np.roll(low_1d, 1) - low_1d
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate True Range
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR (using Wilder's smoothing, equivalent to EWMA with alpha=1/period)
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period] = np.mean(tr[:atr_period+1])  # seed with simple average
    for i in range(atr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/atr_period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/atr_period, adjust=False).mean().values / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/atr_period, adjust=False).mean().values
    
    # Align 1d indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # reuse volume MA
    volume_spike_1d = volume > (2.0 * volume_spike_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX, 20 for volume MA)
    warmup = 70
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r3_level = camarilla_r3[i]
        s3_level = camarilla_s3[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # convert back to boolean
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Camarilla S3 or ADX drops below 20 (trend weakening)
            if price < s3_level or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Camarilla R3 or ADX drops below 20 (trend weakening)
            if price > r3_level or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND ADX > 25 (strong trend) AND volume spike
            if price > r3_level and adx_val > 25 and vol_spike:
                signals[i] = 0.30
                position = 1
            
            # SHORT: Price breaks below Camarilla S3 AND ADX > 25 (strong trend) AND volume spike
            elif price < s3_level and adx_val > 25 and vol_spike:
                signals[i] = -0.30
                position = -1
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "12h_Camarilla_R3_S3_1dADX_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0