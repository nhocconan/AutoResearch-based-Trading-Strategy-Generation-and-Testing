#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1d ADX trend filter
# Camarilla levels from daily data provide intraday support/resistance with high probability reversal/breakout points
# Volume spike (2.0x 50-period average) confirms institutional participation
# ADX > 25 on daily timeframe ensures we only trade in trending markets, reducing whipsaws
# Works in bull markets via breakouts above R3 and bear markets via breakdowns below S3
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1dVolumeSpike_ADXTrend_v1"
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
    
    # Load daily data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: based on previous day's range
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_r3 = camarilla_pivot + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = camarilla_pivot - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate ADX on daily timeframe for trend filter
    # ADX calculation: +DI, -DI, then DX, then smoothed ADX
    period = 14
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed TR, +DM, -DM
    tr_period = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    plus_dm_period = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_period = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 50-period average on daily
    vol_ma_50 = pd.Series(df_1d['volume']).rolling(window=50, min_periods=50).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ma_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and ADX > 25 (trending market)
            if curr_volume_spike and curr_adx > 25:
                # Bullish entry: break above Camarilla R3
                if curr_high > curr_r3 and curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below Camarilla S3
                elif curr_low < curr_s3 and curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Camarilla S3 (reversal signal)
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Camarilla R3 (reversal signal)
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals