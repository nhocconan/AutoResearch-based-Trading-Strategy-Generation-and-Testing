#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and 1d volume spike confirmation.
# Long when price breaks above R3 with 1d ADX > 25 (strong trend) and 1d volume > 2.0x 20-period average.
# Short when price breaks below S3 with 1d ADX > 25 and 1d volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses discrete position sizing (0.25) to balance return and risk. Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe.
# Works in bull/bear: 1d ADX ensures we only trade strong trends, Camarilla R3/S3 provides wide structure to avoid whipsaws, volume spike confirms institutional participation.

name = "12h_Camarilla_R3S3_Breakout_1dADX_1dVolumeSpike"
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
    
    # --- 12h volume confirmation: > 2.0x 20-period average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX (14-period)
    adx_period = 14
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).fillna(0).values
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0).values
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0).values
    # Smoothed TR, DM+
    tr_period = pd.Series(tr).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    # DI+, DI-, DX
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    # ADX
    adx = pd.Series(dx).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    adx_strong = adx > 25.0
    
    # 1d volume confirmation: > 2.0x 20-period average (volume spike)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align 1d indicators to 12h
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong.astype(float))
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # --- 1d Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) > 0:
        # Map each 12h bar to prior day's OHLC
        open_time = prices['open_time']
        prior_day_start = open_time - pd.Timedelta(days=1)
        prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
        
        # Create series of prior day data aligned to 12h bars
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
        if (np.isnan(adx_strong_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(volume_confirm_12h[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + 1d ADX strong trend + 1d volume spike + 12h volume confirmation
            if (close[i] > camarilla_r3[i] and 
                adx_strong_aligned[i] > 0.5 and 
                volume_confirm_1d_aligned[i] > 0.5 and
                volume_confirm_12h[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 1d ADX strong trend + 1d volume spike + 12h volume confirmation
            elif (close[i] < camarilla_s3[i] and 
                  adx_strong_aligned[i] > 0.5 and 
                  volume_confirm_1d_aligned[i] > 0.5 and
                  volume_confirm_12h[i] > 0.5):
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