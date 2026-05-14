#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and 12h volume spike (>1.8x 20-period average).
# Williams %R measures overbought/oversold levels. Long when %R < -80 (oversold) AND close > 1d EMA50 (bullish trend) AND volume > 1.8x MA20.
# Short when %R > -20 (overbought) AND close < 1d EMA50 (bearish trend) AND volume > 1.8x MA20.
# Exit when %R crosses above -50 (for long) or below -50 (for short) OR price crosses 1d EMA50 in opposite direction.
# Uses 1d HTF for trend to reduce noise and overtrading. Volume confirmation (>1.8x) filters weak breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe.
# Williams %R is effective in ranging markets and captures reversals in both bull and bear regimes when combined with trend filter.

name = "12h_WilliamsR_MeanReversion_1dEMA50_12hVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # 12h volume confirmation: > 1.8x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND close > 1d EMA50 (bullish trend) AND volume confirm
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND close < 1d EMA50 (bearish trend) AND volume confirm
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (exiting oversold) OR close < 1d EMA50 (trend change)
            if (williams_r[i] > -50 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (exiting overbought) OR close > 1d EMA50 (trend change)
            if (williams_r[i] < -50 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals