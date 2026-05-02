#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w ADX trend filter
# Camarilla R3/S3 levels act as strong intraday support/resistance; breakouts with volume confirm institutional participation
# 1w ADX > 25 ensures we only trade in strong weekly trends, reducing whipsaws in ranging markets
# Discrete position sizing (0.25) balances profit potential with fee drag control
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns
# Works in bull markets by catching breakouts with trend, works in bear by only taking trend-aligned breaks
# Focus on BTC/ETH as primary symbols

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wADX_Trend"
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
    
    # 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ema_20_1d)  # Volume > 2x 20-period EMA
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(low_1w)
    tr = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(0, high_1w[i] - high_1w[i-1])
        minus_dm[i] = max(0, low_1w[i-1] - low_1w[i])
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    tr_sum = np.zeros_like(tr)
    plus_dm_sum = np.zeros_like(plus_dm)
    minus_dm_sum = np.zeros_like(minus_dm)
    
    tr_sum[period-1] = np.nansum(tr[1:period+1])
    plus_dm_sum[period-1] = np.nansum(plus_dm[1:period+1])
    minus_dm_sum[period-1] = np.nansum(minus_dm[1:period+1])
    
    for i in range(period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i]
    
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = np.zeros_like(tr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX is smoothed DX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.nanmean(dx[period:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_trend = adx > 25  # ADX > 25 indicates strong trend
    adx_trend_aligned = align_htf_to_ltf(prices, df_1w, adx_trend)
    
    # 12h Camarilla levels (based on previous day's OHLC)
    # We need to calculate Camarilla levels for each 12h bar using prior 1d data
    # Since we're on 12h timeframe, we use the prior 1d bar's OHLC
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_for_camarilla) < 2:
        return np.zeros(n)
    
    # For each 12h bar, we use the prior completed 1d bar's OHLC
    high_1d = df_1d_for_camarilla['high'].values
    low_1d = df_1d_for_camarilla['low'].values
    close_1d = df_1d_for_camarilla['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # For each 12h bar, find the prior 1d bar's OHLC
    # We need to align 1d data to 12h bars: each 12h bar sees the prior 1d bar's close
    close_1d_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, close_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, low_1d)
    
    # Calculate Camarilla levels for each bar using prior 1d OHLC
    for i in range(n):
        if i == 0:  # No prior data
            continue
        # Use prior 1d bar's OHLC (available at bar i-1 in 1d timeframe)
        # Since we're using aligned arrays, we can use the value at i-1
        if i-1 >= 0 and not np.isnan(close_1d_aligned[i-1]) and not np.isnan(high_1d_aligned[i-1]) and not np.isnan(low_1d_aligned[i-1]):
            H = high_1d_aligned[i-1]
            L = low_1d_aligned[i-1]
            C = close_1d_aligned[i-1]
            camarilla_r3[i] = C + (H - L) * 1.1 / 4
            camarilla_s3[i] = C - (H - L) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, 2*14)  # ADX needs 2*period for first value
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(adx_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Camarilla R3 with volume spike and bullish trend (ADX > 25)
            if close[i] > camarilla_r3[i] and volume_spike_1d_aligned[i] and adx_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 with volume spike and bullish trend (ADX > 25)
            elif close[i] < camarilla_s3[i] and volume_spike_1d_aligned[i] and adx_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Camarilla S3 (reversal to S3 level)
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Camarilla R3 (reversal to R3 level)
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals