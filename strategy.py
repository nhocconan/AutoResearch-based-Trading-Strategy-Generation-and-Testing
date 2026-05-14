#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above R3 AND close > 4h EMA50 AND 1d volume > 1.8 * 20-period average volume.
# Short when price breaks below S3 AND close < 4h EMA50 AND 1d volume > 1.8 * 20-period average volume.
# Exit when price retouches the 4h EMA50 (mean reversion to trend).
# Uses discrete position sizing (0.20) to limit fee churn. Designed for BTC/ETH robustness by capturing institutional breakouts with volume confirmation in trending markets.
# Target: 80-140 total trades over 4 years (20-35/year) for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Camarilla pivot points (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1h bar using prior day's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # For each 1h bar, use prior completed day's OHLC
    for i in range(n):
        # Get the date of current 1h bar
        current_time = prices.iloc[i]['open_time']
        # Find prior day's OHLC (we need to look back to previous day's close)
        prior_day_start = current_time.normalize() - pd.Timedelta(days=1)
        prior_day_end = prior_day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Get prior day's OHLC from 1d dataframe
        day_mask = (df_1d_ohlc['open_time'] >= prior_day_start) & (df_1d_ohlc['open_time'] <= prior_day_end)
        if day_mask.any():
            prior_day = df_1d_ohlc.loc[day_mask].iloc[0]
            high_prior = prior_day['high']
            low_prior = prior_day['low']
            close_prior = prior_day['close']
            
            # Camarilla calculations
            range_prior = high_prior - low_prior
            camarilla_r3[i] = close_prior + range_prior * 1.1 / 4
            camarilla_s3[i] = close_prior - range_prior * 1.1 / 4
        else:
            # Not enough data, use NaN
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to ensure we have prior data
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices.iloc[i]['open_time']).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND close > 4h EMA50 AND volume spike
            if (open_[i] <= camarilla_r3[i] and close[i] > camarilla_r3[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S3 AND close < 4h EMA50 AND volume spike
            elif (open_[i] >= camarilla_s3[i] and close[i] < camarilla_s3[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retouches 4h EMA50 (mean reversion to trend)
            if close[i] <= ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price retouches 4h EMA50 (mean reversion to trend)
            if close[i] >= ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals