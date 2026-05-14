#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above R3 AND 1d EMA34 > EMA200 (bullish trend) AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below S3 AND 1d EMA34 < EMA200 (bearish trend) AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the prior day's close (Camarilla pivot point).
# Uses discrete position sizing (0.25) to limit fee churn. Target: 75-200 total trades over 4 years (19-50/year) for 4h.
# Works in both bull and bear markets: 1d EMA crossover filter ensures we only trade in clear trending conditions,
# while volume confirmation avoids breakouts in low-participation environments.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34Trend_1dVolumeConfirm_v1"
timeframe = "4h"
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
    
    # Calculate 1d EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA34 and EMA200 on 1d timeframe
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Bullish trend: EMA34 > EMA200, Bearish trend: EMA34 < EMA200
    ema_trend_bullish = ema34_1d > ema200_1d
    ema_trend_bearish = ema34_1d < ema200_1d
    
    # Calculate 1d volume confirmation filter
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align to 4h timeframe
    ema_trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_bullish.astype(float))
    ema_trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_bearish.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Camarilla pivot points (based on previous day's OHLC)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_cp = np.full(n, np.nan)  # Pivot point (close of prior day)
    
    # For each 4h bar, use prior completed day's OHLC
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        prior_day_start = current_time.normalize() - pd.Timedelta(days=1)
        prior_day_end = prior_day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        day_mask = (df_1d['open_time'] >= prior_day_start) & (df_1d['open_time'] <= prior_day_end)
        if day_mask.any():
            prior_day = df_1d.loc[day_mask].iloc[0]
            high_prior = prior_day['high']
            low_prior = prior_day['low']
            close_prior = prior_day['close']
            
            range_prior = high_prior - low_prior
            camarilla_r3[i] = close_prior + range_prior * 1.1 / 4  # R3 level
            camarilla_s3[i] = close_prior - range_prior * 1.1 / 4  # S3 level
            camarilla_cp[i] = close_prior  # Camarilla pivot point is the prior day's close
        else:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_cp[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_trend_bullish_aligned[i]) or 
            np.isnan(ema_trend_bearish_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_cp[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND bullish 1d EMA trend AND volume confirmation
            if (open_[i] <= camarilla_r3[i] and close[i] > camarilla_r3[i] and 
                ema_trend_bullish_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 AND bearish 1d EMA trend AND volume confirmation
            elif (open_[i] >= camarilla_s3[i] and close[i] < camarilla_s3[i] and 
                  ema_trend_bearish_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
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