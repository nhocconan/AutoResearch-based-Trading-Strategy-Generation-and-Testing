#!/usr/bin/env python3
# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation. 
# Long when RSI < 30, 4h close > 4h EMA50 (uptrend), and 4h volume > 1.5x 20-period average. 
# Short when RSI > 70, 4h close < 4h EMA50 (downtrend), and 4h volume > 1.5x 20-period average. 
# Exit on opposite RSI extreme (RSI > 70 for longs, RSI < 30 for shorts) or at Camarilla pivot point (prior day close).
# Uses 08-20 UTC session filter to avoid low-volume periods. Position size fixed at 0.20 to limit fee churn.
# Target: 60-150 trades over 4 years (15-37/year) for 1h timeframe.
# Works in bull/bear: 4h EMA50 ensures trend alignment, RSI provides mean-reversion entries within trend.

name = "1h_RSI_MeanReversion_4hEMA50_Trend_Volume_Session"
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
    
    # --- 1h RSI (14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Camarilla Pivot Point (Prior Day Close) ---
    camarilla_cp = np.full(n, np.nan)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        # Map each 1h bar to prior day's close
        open_time = prices['open_time']
        prior_day_close = open_time - pd.Timedelta(days=1)
        prior_day_close = prior_day_close.dt.normalize()  # Start of prior day
        
        # Create series of prior day closes aligned to 1h bars
        prior_day_close_series = pd.Series(index=open_time, dtype='datetime64[ns]')
        for i in range(n):
            pd_ts = prior_day_close.iloc[i]
            day_mask = (df_1d['open_time'] >= pd_ts) & (df_1d['open_time'] < pd_ts + pd.Timedelta(days=1))
            if day_mask.any():
                camarilla_cp[i] = df_1d.loc[day_mask, 'close'].iloc[0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if outside session or missing data
        if (not in_session[i] or
            np.isnan(ema_50_bullish_aligned[i]) or 
            np.isnan(ema_50_bearish_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(camarilla_cp[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI < 30 (oversold) + 4h uptrend + volume confirmation
            if (rsi[i] < 30 and 
                ema_50_bullish_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI > 70 (overbought) + 4h downtrend + volume confirmation
            elif (rsi[i] > 70 and 
                  ema_50_bearish_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 70 (overbought) OR price >= Camarilla pivot (prior day close)
            if rsi[i] > 70 or close[i] >= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI < 30 (oversold) OR price <= Camarilla pivot (prior day close)
            if rsi[i] < 30 or close[i] <= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals