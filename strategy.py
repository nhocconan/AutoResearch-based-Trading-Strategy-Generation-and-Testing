#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 with 1d EMA34 uptrend and 12h volume > 1.5x 20-period average.
# Short when price breaks below S3 with 1d EMA34 downtrend and 12h volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or at prior day close (CP).
# Uses 00-24 UTC session (no session filter) to maximize trades in 12h timeframe.
# Position size fixed at 0.25 to balance profit and fee drag. Target: 80-120 trades over 4 years (20-30/year).
# Works in bull/bear: 1d EMA34 ensures trend alignment, Camarilla provides structure within trend.
# Uses discrete position sizing to minimize fee churn and volume confirmation to reduce false breakouts.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # --- 12h Indicators (LTF) ---
    # 12h Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_bullish = close_1d > ema_34  # Bullish if price above EMA34
    ema_34_bearish = close_1d < ema_34  # Bearish if price below EMA34
    
    # Align 1d indicators to 12h
    ema_34_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_34_bullish.astype(float))
    ema_34_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_34_bearish.astype(float))
    
    # --- 12h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_cp = np.full(n, np.nan)  # Prior day close
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
                camarilla_cp[i] = close_val
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_bullish_aligned[i]) or 
            np.isnan(ema_34_bearish_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_cp[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + 1d uptrend + volume confirmation
            if (close[i] > camarilla_r3[i] and 
                ema_34_bullish_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 1d downtrend + volume confirmation
            elif (close[i] < camarilla_s3[i] and 
                  ema_34_bearish_aligned[i] > 0.5 and 
                  volume_confirm[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR price <= prior day close (CP)
            if close[i] < camarilla_s3[i] or close[i] <= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR price >= prior day close (CP)
            if close[i] > camarilla_r3[i] or close[i] >= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals