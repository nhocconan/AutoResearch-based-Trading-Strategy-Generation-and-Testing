#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation.
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed to capture medium-term
# reversals at key pivot levels in both bull and bear markets by combining Camarilla structure,
# 1w trend filter, and volume strength. Targets 30-100 total trades over 4 years (7-25/year).

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # Volume spike: > 2.0x 20-period average (strict threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Camarilla R3 and S3 levels from prior bar
    # R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close + 1.1 * (high - low) / 4
    camarilla_s3 = close - 1.1 * (high - low) / 4
    # Shift by 1 to use prior bar's levels (no look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) - trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND close > 1w EMA34 (bullish trend) AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 AND close < 1w EMA34 (bearish trend) AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1w EMA34 (trend change) OR touches Camarilla S3 (mean reversion)
            if close[i] < ema_34_1w_aligned[i] or close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1w EMA34 (trend change) OR touches Camarilla R3 (mean reversion)
            if close[i] > ema_34_1w_aligned[i] or close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals