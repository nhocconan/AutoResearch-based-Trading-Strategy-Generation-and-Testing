#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 4h volume spike confirmation.
# Long when price breaks above R1 AND 4h EMA50 slope is positive AND 4h volume > 1.8 * 20-period average.
# Short when price breaks below S1 AND 4h EMA50 slope is negative AND 4h volume > 1.8 * 20-period average.
# Exit when price retraces to the prior day's close (Camarilla pivot point).
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year) for 1h.
# Works in both bull and bear markets: 4h EMA50 slope filter ensures we only trade in clear trending conditions,
# while volume confirmation avoids breakouts in low-participation environments. Session filter (08-20 UTC) reduces noise.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_Volume_Session"
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
    
    # Precompute session hours (08-20 UTC) to avoid per-bar datetime ops
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # EMA50
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA50 slope: bullish if current > previous, bearish if current < previous
    ema_50_slope = np.diff(ema_50, prepend=ema_50[0])
    ema_50_bullish = ema_50_slope > 0
    ema_50_bearish = ema_50_slope < 0
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume_4h > (1.8 * vol_ma_20_4h)
    
    # Align to 1h timeframe
    ema_50_bullish_aligned = align_htf_to_ltf(prices, df_4h, ema_50_bullish.astype(float))
    ema_50_bearish_aligned = align_htf_to_ltf(prices, df_4h, ema_50_bearish.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_4h, volume_confirm_4h.astype(float))
    
    # Calculate Camarilla pivot points (based on previous day's OHLC)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_cp = np.full(n, np.nan)  # Pivot point (close of prior day)
    
    # For each 1h bar, use prior completed day's OHLC
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        prior_day_start = current_time.normalize() - pd.Timedelta(days=1)
        prior_day_end = prior_day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Get HTF 1d data for pivot calculation
        df_1d = get_htf_data(prices, '1d')
        day_mask = (df_1d['open_time'] >= prior_day_start) & (df_1d['open_time'] <= prior_day_end)
        if day_mask.any():
            prior_day = df_1d.loc[day_mask].iloc[0]
            high_prior = prior_day['high']
            low_prior = prior_day['low']
            close_prior = prior_day['close']
            
            range_prior = high_prior - low_prior
            camarilla_r1[i] = close_prior + range_prior * 1.1 / 2  # R1 level
            camarilla_s1[i] = close_prior - range_prior * 1.1 / 2  # S1 level
            camarilla_cp[i] = close_prior  # Camarilla pivot point is the prior day's close
        else:
            camarilla_r1[i] = np.nan
            camarilla_s1[i] = np.nan
            camarilla_cp[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN or outside session
        if (not in_session[i] or
            np.isnan(ema_50_bullish_aligned[i]) or 
            np.isnan(ema_50_bearish_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i]) or
            np.isnan(camarilla_cp[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 AND bullish 4h EMA50 trend AND volume confirmation
            if (open_[i] <= camarilla_r1[i] and close[i] > camarilla_r1[i] and 
                ema_50_bullish_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S1 AND bearish 4h EMA50 trend AND volume confirmation
            elif (open_[i] >= camarilla_s1[i] and close[i] < camarilla_s1[i] and 
                  ema_50_bearish_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Camarilla pivot point (CP)
            if close[i] <= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price retraces to Camarilla pivot point (CP)
            if close[i] >= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals