#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation.
# Long when price breaks above R1 with 4h EMA50 uptrend and 4h volume > 1.5x 20-period average.
# Short when price breaks below S1 with 4h EMA50 downtrend and 4h volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or at prior day close (CP).
# Uses 08-20 UTC session filter to avoid low-volume periods. Position size fixed at 0.20 to limit fee churn.
# Target: 60-150 trades over 4 years (15-37/year) for 1h timeframe.
# Works in bull/bear: 4h EMA50 ensures trend alignment, Camarilla provides structure within trend.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_Volume_Session"
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
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA50 for trend
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_bullish = close_4h > ema_50  # Bullish if price above EMA50
    ema_50_bearish = close_4h < ema_50  # Bearish if price below EMA50
    
    # 4h Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_4h > (1.5 * vol_ma_20)
    
    # Align 4h indicators to 1h
    ema_50_bullish_aligned = align_htf_to_ltf(prices, df_4h, ema_50_bullish.astype(float))
    ema_50_bearish_aligned = align_htf_to_ltf(prices, df_4h, ema_50_bearish.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_4h, volume_confirm.astype(float))
    
    # --- 1h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_cp = np.full(n, np.nan)  # Prior day close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        # Map each 1h bar to prior day's OHLC
        open_time = prices['open_time']
        prior_day_start = open_time - pd.Timedelta(days=1)
        prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
        
        # Create series of prior day data aligned to 1h bars
        for i in range(n):
            pd_ts = prior_day_start.iloc[i]
            day_mask = (df_1d['open_time'] >= pd_ts) & (df_1d['open_time'] < pd_ts + pd.Timedelta(days=1))
            if day_mask.any():
                day_data = df_1d.loc[day_mask]
                high_val = day_data['high'].iloc[0]
                low_val = day_data['low'].iloc[0]
                close_val = day_data['close'].iloc[0]
                range_val = high_val - low_val
                camarilla_r1[i] = close_val + (range_val * 1.1 / 12)
                camarilla_s1[i] = close_val - (range_val * 1.1 / 12)
                camarilla_cp[i] = close_val
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if outside session or missing data
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
            # LONG: Price breaks above R1 + 4h uptrend + volume confirmation
            if (close[i] > camarilla_r1[i] and 
                ema_50_bullish_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + 4h downtrend + volume confirmation
            elif (close[i] < camarilla_s1[i] and 
                  ema_50_bearish_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR price <= prior day close (CP)
            if close[i] < camarilla_s1[i] or close[i] <= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR price >= prior day close (CP)
            if close[i] > camarilla_r1[i] or close[i] >= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals