#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and 1d volume confirmation.
# Long when price breaks above R3 with 4h EMA34 uptrend and 1d volume > 1.5x 20-period average.
# Short when price breaks below S3 with 4h EMA34 downtrend and 1d volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts).
# Session filter: only trade 08-20 UTC to avoid low-liquidity periods.
# Uses discrete position sizing (0.20) to minimize fee churn and strict volume confirmation to reduce false breakouts.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.
# Works in bull/bear: 4h EMA34 ensures trend alignment, Camarilla provides structure within trend, volume confirms institutional participation.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_1dVolume_Session"
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
    
    # --- 1h Indicators (LTF) ---
    # 1h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1h = volume > (1.5 * vol_ma_20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA34 for trend
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_bullish = close_4h > ema_34  # Bullish if price above EMA34
    ema_34_bearish = close_4h < ema_34  # Bearish if price below EMA34
    
    # Align 4h indicators to 1h
    ema_34_bullish_aligned = align_htf_to_ltf(prices, df_4h, ema_34_bullish.astype(float))
    ema_34_bearish_aligned = align_htf_to_ltf(prices, df_4h, ema_34_bearish.astype(float))
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    
    # Align 1d volume confirmation to 1h
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # --- 1h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) > 0:
        # Map each 1h bar to prior day's OHLC
        open_time = prices['open_time']
        prior_day_start = open_time - pd.Timedelta(days=1)
        prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
        
        # Create series of prior day data aligned to 1h bars
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_bullish_aligned[i]) or 
            np.isnan(ema_34_bearish_aligned[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(volume_confirm_1h[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + 4h EMA34 uptrend + 1d volume confirmation + 1h volume confirmation
            if (close[i] > camarilla_r3[i] and 
                ema_34_bullish_aligned[i] > 0.5 and 
                volume_confirm_1d_aligned[i] > 0.5 and
                volume_confirm_1h[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 + 4h EMA34 downtrend + 1d volume confirmation + 1h volume confirmation
            elif (close[i] < camarilla_s3[i] and 
                  ema_34_bearish_aligned[i] > 0.5 and 
                  volume_confirm_1d_aligned[i] > 0.5 and
                  volume_confirm_1h[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R3
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals