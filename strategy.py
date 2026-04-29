#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout with 1d ADX25 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R3 AND 1d ADX > 25 (trending) AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 AND 1d ADX > 25 (trending) AND volume > 2.0x 20-bar avg
# Exit when price retests Camarilla pivot (PP) or opposite level (S3 for long, R3 for short)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# ADX filter ensures we only trade in trending markets, reducing whipsaws in ranging conditions.
# Camarilla R3/S3 are stronger levels than R1/S1, reducing false breakouts.
# Volume confirmation ensures breakout strength. Works in bull via breakout continuation,
# in bear via breakdown continuation. Novelty: using ADX trend filter instead of EMA for better regime adaptation.

name = "12h_Camarilla_R3S3_Breakout_1dADX25_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar (to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first bar to NaN (no previous bar)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4.0
    
    # Calculate 1d ADX for trend filter (ADX > 25 = trending market)
    # ADX calculation: +DI, -DI, then DX, then ADX
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * EMA(+DM, 14) / EMA(TR, 14)
    # -DI = 100 * EMA(-DM, 14) / EMA(TR, 14)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EMA(DX, 14)
    
    # Calculate True Range (TR)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Calculate Directional Movement (+DM, -DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Set first values to NaN
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan
    
    # Calculate smoothed averages using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Calculate Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: prev * (1 - 1/period) + current * (1/period)
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
                else:
                    result[i] = np.nan
        return result
    
    # Smooth TR, +DM, -DM over 14 periods
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    # Calculate DX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Calculate ADX (smoothed DX)
    adx_14 = wilders_smoothing(dx, 14)
    
    # Align all 1d indicators to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and ADX warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_pp = camarilla_pp_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_adx_1d = adx_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot (PP) or breaks below S3
            if curr_low <= curr_pp or curr_close <= curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot (PP) or breaks above R3
            if curr_high >= curr_pp or curr_close >= curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND ADX > 25 (trending) AND volume confirmation
            if curr_high > curr_r3 and curr_adx_1d > 25.0 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND ADX > 25 (trending) AND volume confirmation
            elif curr_low < curr_s3 and curr_adx_1d > 25.0 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals