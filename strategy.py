#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and 1d volume spike confirmation.
# Uses discrete position sizing (0.0, ±0.20) to minimize fee churn. Designed to capture medium-term reversals
# at key pivot levels in both bull and bear markets by combining Camarilla structure, 4h trend filter,
# and 1d volume strength. Targets 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Uses session filter (08-20 UTC) to reduce noise trades.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_1dVolumeSpike_v1"
timeframe = "1h"
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
    
    # Pre-compute session hours (08-20 UTC) to reduce noise trades
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 1h Indicators (LTF) ---
    # Camarilla R3 and S3 levels from prior bar
    # R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close + 1.1 * (high - low) / 4
    camarilla_s3 = close - 1.1 * (high - low) / 4
    # Shift by 1 to use prior bar's levels (no look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA(34) - trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 1d volume spike: > 2.0x 20-period average (strict threshold to reduce trades)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND close > 4h EMA34 (bullish trend) AND 1d volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S3 AND close < 4h EMA34 (bearish trend) AND 1d volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 4h EMA34 (trend change) OR touches Camarilla S3 (mean reversion)
            if close[i] < ema_34_4h_aligned[i] or close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above 4h EMA34 (trend change) OR touches Camarilla R3 (mean reversion)
            if close[i] > ema_34_4h_aligned[i] or close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals