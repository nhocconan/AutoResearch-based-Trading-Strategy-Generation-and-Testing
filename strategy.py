#!/usr/bin/env python3
# Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA34 trend filter and 6h volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND close > 1d EMA34 (bullish trend) AND volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) AND close < 1d EMA34 (bearish trend) AND volume > 2.0x 20-period average.
# Exit when Williams %R returns to -50 (mean reversion) OR opposite extreme is touched.
# Uses 1d HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (2.0x) reduces false signals.
# Williams %R is a momentum oscillator that identifies overbought/oversold conditions, effective in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.

name = "6h_WilliamsR_Extreme_1dEMA34_6hVolumeConfirm_v1"
timeframe = "6h"
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
    
    # --- 6h Williams %R (14-period) ---
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # --- 6h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades) ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND close > 1d EMA34 (bullish trend) AND volume confirm
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND close < 1d EMA34 (bearish trend) AND volume confirm
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 (mean reversion) OR touches -20 (overbought)
            if (williams_r[i] >= -50 or 
                williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 (mean reversion) OR touches -80 (oversold)
            if (williams_r[i] <= -50 or 
                williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals