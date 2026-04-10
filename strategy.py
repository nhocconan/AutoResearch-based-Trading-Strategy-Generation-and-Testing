#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# - Camarilla levels (R3/R4, S3/S4) from 1d OHLC for institutional breakout/fade zones
# - Long when price breaks above R4 with 1d ADX > 25 (strong uptrend) and volume spike
# - Short when price breaks below S3 with 1d ADX > 25 (strong downtrend) and volume spike
# - Exit: price returns to 1d VWAP or opposite Camarilla level (R3/S4)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# - Works in bull/bear: breakouts capture trends, ADX filter avoids whipsaws in ranging markets

name = "6h_1d_camarilla_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    #          S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    #          where C = (H+L+C)/3 (typical price), but standard uses close
    # Actually, standard Camarilla uses: R4 = Close + ((High-Low) * 1.1/2)
    # We'll use previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (already aligned by get_htf_data shift)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d ADX for trend filter
    # ADX calculation: +DM, -DM, TR, then DX, then ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            smoothed[period-1] = np.nanmean(values[:period])
            # Subsequent values: smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
            for i in range(period, len(values)):
                smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        return smoothed
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d VWAP for exit reference
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = np.cumsum(typical_price_1d * df_1d['volume'].values) / np.cumsum(df_1d['volume'].values)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate volume spike detector (6h volume > 2x 20-period SMA)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_sma_20)
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vwap = vwap_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        long_breakout = curr_close > r4 and adx_val > 25 and vol_spike
        short_breakout = curr_low < s3 and adx_val > 25 and vol_spike
        
        # Exit conditions: price returns to VWAP or opposite Camarilla level
        long_exit = curr_close < vwap or curr_close < r3
        short_exit = curr_close > vwap or curr_close > s4
        
        if position == 0:  # Flat - look for entry
            if long_breakout:
                position = 1
                signals[i] = 0.25
            elif short_breakout:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals