#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from oversold with 1d ADX > 25 and 6h volume > 1.5x 20-period average.
# Short when Williams %R crosses below -20 from overbought with 1d ADX > 25 and 6h volume > 1.5x 20-period average.
# Exit when Williams %R returns to opposite extreme (-20 for longs, -80 for shorts) or at 6h close below/above 20-period EMA.
# Uses 08-20 UTC session filter to focus on liquid hours. Position size fixed at 0.25 to balance return and drawdown.
# Williams %R captures mean reversals in ranging markets while ADX ensures we only trade during trending regimes.
# Target: 80-160 total trades over 4 years (20-40/year) for 6h timeframe.

name = "6h_WilliamsR_Reversal_1dADXTrend_1dVolumeConfirm_v1"
timeframe = "6h"
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
    
    # --- 6h Indicators (LTF) ---
    # 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 6h Williams %R cross above -80 (long signal) and below -20 (short signal)
    williams_r_long = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_r_short = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    # Handle first element
    williams_r_long[0] = False
    williams_r_short[0] = False
    
    # 6h Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # 6h EMA20 for exit condition
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX (14-period) for trend strength
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    
    adx_trend = adx > 25  # Strong trend when ADX > 25
    
    # Align 1d indicators to 6h
    adx_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if outside session or missing data
        if (not in_session[i] or
            np.isnan(williams_r[i]) or 
            np.isnan(adx_trend_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 + ADX trend + volume confirmation
            if (williams_r_long[i] and 
                adx_trend_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 + ADX trend + volume confirmation
            elif (williams_r_short[i] and 
                  adx_trend_aligned[i] > 0.5 and 
                  volume_confirm[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 OR close below EMA20
            if williams_r[i] >= -20 or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 OR close above EMA20
            if williams_r[i] <= -80 or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals