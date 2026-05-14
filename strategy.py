#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 with 1w EMA34 uptrend and 1d volume > 1.8x 20-period average.
# Short when price breaks below S3 with 1w EMA34 downtrend and 1d volume > 1.8x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or at prior week close (weekly CP).
# Uses discrete position sizing (0.25) to limit fee churn and strict volume confirmation to reduce false breakouts.
# Target: 30-80 trades over 4 years (7-20/year) for 1d timeframe.
# Works in bull/bear: 1w EMA34 ensures trend alignment, Camarilla provides structure within trend.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_Volume"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # 1d Volume confirmation: > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_bullish = close_1w > ema_34  # Bullish if price above EMA34
    ema_34_bearish = close_1w < ema_34  # Bearish if price below EMA34
    
    # Align 1w indicators to 1d
    ema_34_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_34_bullish.astype(float))
    ema_34_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_34_bearish.astype(float))
    
    # --- 1d Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_cp = np.full(n, np.nan)  # Prior day close
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Map each 1d bar to prior day's OHLC using vectorized approach
    open_time = prices['open_time']
    prior_day_start = open_time - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
    
    # Create a mapping from date to prior day's OHLC
    df_1d_pivot['date'] = df_1d_pivot['open_time'].dt.normalize()
    prior_day_data = df_1d_pivot.set_index('date')
    
    for i in range(n):
        pd_ts = prior_day_start.iloc[i]
        if pd_ts in prior_day_data.index:
            day_data = prior_day_data.loc[pd_ts]
            high_val = day_data['high']
            low_val = day_data['low']
            close_val = day_data['close']
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
            # LONG: Price breaks above R3 + 1w uptrend + volume confirmation
            if (close[i] > camarilla_r3[i] and 
                ema_34_bullish_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 1w downtrend + volume confirmation
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