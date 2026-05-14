#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above R3 AND 1d EMA34 is bullish AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below S3 AND 1d EMA34 is bearish AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the Camarilla pivot point (prior day's close).
# Uses discrete position sizing (0.25) to balance profit and fee drag. Designed for BTC/ETH robustness by capturing institutional breakouts with volume confirmation in trending markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_1dVolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d volume spike filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Camarilla pivot points (based on previous day's OHLC)
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar using prior day's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_cp = np.full(n, np.nan)  # Pivot point (close of prior day)
    
    # For each 12h bar, use prior completed day's OHLC
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        prior_day_start = current_time.normalize() - pd.Timedelta(days=1)
        prior_day_end = prior_day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        day_mask = (df_1d_ohlc['open_time'] >= prior_day_start) & (df_1d_ohlc['open_time'] <= prior_day_end)
        if day_mask.any():
            prior_day = df_1d_ohlc.loc[day_mask].iloc[0]
            high_prior = prior_day['high']
            low_prior = prior_day['low']
            close_prior = prior_day['close']
            
            range_prior = high_prior - low_prior
            camarilla_r3[i] = close_prior + range_prior * 1.1 / 4
            camarilla_s3[i] = close_prior - range_prior * 1.1 / 4
            camarilla_cp[i] = close_prior  # Camarilla pivot point is the prior day's close
        else:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_cp[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_cp[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = prices.index[i].hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND 1d EMA34 is bullish (price > EMA34) AND volume spike
            if (open_[i] <= camarilla_r3[i] and close[i] > camarilla_r3[i] and 
                close[i] > ema34_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 AND 1d EMA34 is bearish (price < EMA34) AND volume spike
            elif (open_[i] >= camarilla_s3[i] and close[i] < camarilla_s3[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Camarilla pivot point (CP)
            if close[i] <= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Camarilla pivot point (CP)
            if close[i] >= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals