#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h ADX trend filter and volume confirmation
# Uses 4h for signal direction (trend + Camarilla levels) and 1h for precise entry timing
# Volume spike confirms breakout validity. Session filter (08-20 UTC) reduces noise.
# Discrete sizing 0.20 balances risk and minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R3S3_Breakout_4hADX25_VolumeSpike_v1"
timeframe = "1h"
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
    
    # Calculate 4h Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    # Use previous 4h bar's OHLC for current Camarilla levels
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r4 = pivot + (prev_high - prev_low) * 1.1 / 2.0
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    s4 = pivot - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r4_aligned = align_htf_to_ltf(prices, df_4h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_4h, s4)
    
    # Calculate 4h ADX(14) for trend filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = np.abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where(
        (df_4h['high'] - df_4h['high'].shift(1)) > (df_4h['low'].shift(1) - df_4h['low']),
        np.maximum(df_4h['high'] - df_4h['high'].shift(1), 0),
        0
    )
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where(
        (df_4h['low'].shift(1) - df_4h['low']) > (df_4h['high'] - df_4h['high'].shift(1)),
        np.maximum(df_4h['low'].shift(1) - df_4h['low'], 0),
        0
    )
    
    # Smoothed +DM and -DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 4h ADX to 1h timeframe (wait for completed 4h bar)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # warmup for volume MA and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r4 = r4_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish breakout: price breaks above R4
                if curr_close > curr_r4:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below S4
                elif curr_close < curr_s4:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R3 (mean reversion)
            if curr_close < curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above S3 (mean reversion)
            if curr_close > curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals