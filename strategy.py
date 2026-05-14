#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation (>1.5x 20-period average).
# Long when price breaks above R3 AND close > 1d EMA34 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below S3 AND close < 1d EMA34 (bearish trend) AND volume > 1.5x MA20.
# Exit when price reverts to the 1d VWAP (mean reversion to daily fair value).
# Camarilla levels from 1d provide intraday support/resistance; 1d EMA34 filters for higher-timeframe trend;
# volume confirmation reduces false breakouts. Works in bull (breakouts continue) and bear (breakdowns continue).
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # --- 6h Indicators (LTF) ---
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Extract 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: R3, S3, R4, S4
    # Camarilla: R4 = close + ((high-low) * 1.1/2), R3 = close + ((high-low) * 1.1/4)
    #          S3 = close - ((high-low) * 1.1/4), S4 = close - ((high-low) * 1.1/2)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + (camarilla_range * 1.1 / 4)
    s3 = close_1d - (camarilla_range * 1.1 / 4)
    r4 = close_1d + (camarilla_range * 1.1 / 2)
    s4 = close_1d - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d VWAP for exit (volume-weighted average price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (typical_price_1d * df_1d['volume'].values).cumsum() / df_1d['volume'].values.cumsum()
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or
            np.isnan(volume_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND bullish trend AND volume confirmation
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND bearish trend AND volume confirmation
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to 1d VWAP (mean reversion) OR breaks below S3 (failed breakout)
            if (close[i] <= vwap_1d_aligned[i] or 
                close[i] < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to 1d VWAP (mean reversion) OR breaks above R3 (failed breakdown)
            if (close[i] >= vwap_1d_aligned[i] or 
                close[i] > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals