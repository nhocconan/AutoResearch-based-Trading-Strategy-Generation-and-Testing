#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 1d ADX trend filter
# Camarilla levels provide mathematically derived support/resistance from prior day.
# Breakout at R3/S3 with volume confirmation indicates institutional participation.
# 1d ADX > 25 ensures we only trade in trending markets, reducing false breakouts.
# Works in bull (breakouts with volume) and bear (trend continuation after pullbacks to levels).
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3S3_Breakout_VolumeSpike_1dADX25_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla levels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels (based on prior 1d candle)
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + camarilla_range * 1.1 / 4
    camarilla_s3 = prev_close - camarilla_range * 1.1 / 4
    camarilla_r4 = prev_close + camarilla_range * 1.1 / 2
    camarilla_s4 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d volume spike: current volume > 2.0 * 20-period average volume
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (volume_ma_20 * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1d ADX(14) calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder smoothing
    def _wilder_smooth(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        result[period-1] = np.nanmean(x[1:period])
        for i in range(period, len(x)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr_1d = _wilder_smooth(tr, 14)
    plus_di_1d = 100 * _wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * _wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = _wilder_smooth(dx_1d, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 20 for volume MA + 14*3 for ADX
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or np.isnan(r4_1d[i]) or 
            np.isnan(s4_1d[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Breakout conditions (using prior bar to avoid look-ahead)
        breakout_up = curr_close > r3_1d[i-1]  # Break above R3
        breakout_down = curr_close < s3_1d[i-1]  # Break below S3
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike_1d_aligned[i]
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3, volume spike, strong trend
            if breakout_up and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S3, volume spike, strong trend
            elif breakout_down and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S3 or weak trend
            if curr_close < s3_1d[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R3 or weak trend
            if curr_close > r3_1d[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals