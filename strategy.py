#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and 12h volume spike confirmation.
# Long when price breaks above R3 AND 12h EMA34 slope is positive AND 12h volume > 2.0 * 20-period average.
# Short when price breaks below S3 AND 12h EMA34 slope is negative AND 12h volume > 2.0 * 20-period average.
# Exit when price retraces to the prior day's close (Camarilla pivot point).
# Uses discrete position sizing (0.30) to balance risk and reward. Target: 75-200 total trades over 4 years (19-50/year) for 4h.
# Works in both bull and bear markets: 12h EMA34 slope filter ensures we only trade in clear trending conditions,
# while volume confirmation avoids breakouts in low-participation environments. R3/S3 levels provide wider bands than R1/S1
# to reduce false breakouts and lower trade frequency, improving test generalization.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA34_Trend_Volume"
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
    
    # Calculate 12h EMA34 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # EMA34
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    # EMA34 slope: bullish if current > previous, bearish if current < previous
    ema_34_slope = np.diff(ema_34, prepend=ema_34[0])
    ema_34_bullish = ema_34_slope > 0
    ema_34_bearish = ema_34_slope < 0
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume_12h > (2.0 * vol_ma_20_12h)
    
    # Align to 4h timeframe
    ema_34_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_34_bullish.astype(float))
    ema_34_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_34_bearish.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm_12h.astype(float))
    
    # Calculate Camarilla pivot points (based on previous day's OHLC)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_cp = np.full(n, np.nan)  # Pivot point (close of prior day)
    
    # For each 4h bar, use prior completed day's OHLC
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
        if (np.isnan(ema_34_bullish_aligned[i]) or 
            np.isnan(ema_34_bearish_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_cp[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND bullish 12h EMA34 trend AND volume confirmation
            if (open_[i] <= camarilla_r3[i] and close[i] > camarilla_r3[i] and 
                ema_34_bullish_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below S3 AND bearish 12h EMA34 trend AND volume confirmation
            elif (open_[i] >= camarilla_s3[i] and close[i] < camarilla_s3[i] and 
                  ema_34_bearish_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Camarilla pivot point (CP)
            if close[i] <= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price retraces to Camarilla pivot point (CP)
            if close[i] >= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals