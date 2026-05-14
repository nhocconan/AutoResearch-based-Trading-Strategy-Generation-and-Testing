#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 1w volume spike confirmation.
# Long when price breaks above R3 with 1d EMA34 uptrend and 1w volume > 2.5x 20-period average.
# Short when price breaks below S3 with 1d EMA34 downtrend and 1w volume > 2.5x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and strict volume confirmation to reduce false breakouts.
# Target: 100-200 total trades over 4 years = 25-50/year for 4h timeframe.
# Works in bull/bear: 1d EMA34 ensures strong trend alignment, Camarilla R3/S3 provides tight structure within trend, 1w volume spike confirms institutional participation.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_1wVolumeSpike"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA34 trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d EMA34 uptrend/downtrend signals
    ema_34_uptrend = ema_34_1d_aligned > np.roll(ema_34_1d_aligned, 1)
    ema_34_downtrend = ema_34_1d_aligned < np.roll(ema_34_1d_aligned, 1)
    # Handle first value
    ema_34_uptrend[0] = False
    ema_34_downtrend[0] = False
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    
    # 1w volume confirmation: > 2.5x 20-period average (volume spike)
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = volume_1w > (2.5 * vol_ma_20_1w)
    
    # Align 1w volume confirmation to 4h
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w.astype(float))
    
    # --- 4h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) > 0:
        # Map each 4h bar to prior day's OHLC
        open_time = prices['open_time']
        prior_day_start = open_time - pd.Timedelta(days=1)
        prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
        
        # Create series of prior day data aligned to 4h bars
        for i in range(n):
            pd_ts = prior_day_start.iloc[i]
            day_mask = (df_1d_pivot['open_time'] >= pd_ts) & (df_1d_pivot['open_time'] < pd_ts + pd.Timedelta(days=1))
            if day_mask.any():
                day_data = df_1d_pivot.loc[day_mask]
                high_val = day_data['high'].iloc[0]
                low_val = day_data['low'].iloc[0]
                close_val = day_data['close'].iloc[0]
                range_val = high_val - low_val
                camarilla_r3[i] = close_val + (range_val * 1.1 / 4)  # R3
                camarilla_s3[i] = close_val - (range_val * 1.1 / 4)  # S3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_uptrend[i]) or 
            np.isnan(ema_34_downtrend[i]) or
            np.isnan(volume_confirm_1w_aligned[i]) or
            np.isnan(volume_confirm_4h[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + 1d EMA34 uptrend + 1w volume spike + 4h volume confirmation
            if (close[i] > camarilla_r3[i] and 
                ema_34_uptrend[i] and 
                volume_confirm_1w_aligned[i] > 0.5 and
                volume_confirm_4h[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 1d EMA34 downtrend + 1w volume spike + 4h volume confirmation
            elif (close[i] < camarilla_s3[i] and 
                  ema_34_downtrend[i] and 
                  volume_confirm_1w_aligned[i] > 0.5 and
                  volume_confirm_4h[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals