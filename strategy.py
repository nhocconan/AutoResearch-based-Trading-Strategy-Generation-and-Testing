#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and 1d volume spike confirmation.
# Long when Bull Power > 0, ADX > 25 (trending), and 1d volume > 1.5x 20-period average.
# Short when Bear Power < 0, ADX > 25 (trending), and 1d volume > 1.5x 20-period average.
# Exit when Elder Power reverses sign or ADX < 20 (range regime).
# Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear: ADX ensures we only trade strong trends,
# Elder Ray captures trend strength via EMA(13) deviation, volume spike confirms institutional participation.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe.

name = "6h_ElderRay_1dADX_1dVolumeSpike"
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
    
    # --- 6h EMA(13) for Elder Ray ---
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # 1d volume spike: > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * vol_ma_20_1d)
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_13[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray components
        bull_power = close[i] - ema_13[i]
        bear_power = ema_13[i] - close[i]
        
        if position == 0:
            # LONG: Bull Power > 0 + ADX > 25 (strong trend) + 1d volume spike
            if (bull_power > 0 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 + ADX > 25 (strong trend) + 1d volume spike
            elif (bear_power > 0 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 OR ADX < 20 (range) OR volume spike ends
            if (bull_power <= 0 or 
                adx_1d_aligned[i] < 20 or 
                volume_spike_1d_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 OR ADX < 20 (range) OR volume spike ends
            if (bear_power <= 0 or 
                adx_1d_aligned[i] < 20 or 
                volume_spike_1d_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals