#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX trend filter and 1d volume spike confirmation.
# Long when Bull Power > 0, Bear Power < 0, 1d ADX > 25 (strong trend), and 1d volume > 2.0x 20-period average.
# Short when Bull Power < 0, Bear Power > 0, 1d ADX > 25, and 1d volume > 2.0x 20-period average.
# Exit when Elder Ray signals weaken (Bull Power <= 0 for longs, Bear Power <= 0 for shorts) or ADX < 20 (weak trend).
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 80-180 total trades over 4 years = 20-45/year.
# Works in bull/bear: 1d ADX ensures we only trade strong trends, Elder Ray measures bull/bear power behind price moves,
# volume spike confirms institutional participation. Avoids ranging markets where Elder Ray gives false signals.

name = "6h_ElderRay_1dADXTrend_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # 6h EMA13 for Elder Ray calculation (typical setting)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX calculation (standard 14-period)
    adx_period = 14
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).fillna(0).values
    # ATR
    atr = pd.Series(tr).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    # Smoothed DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # 1d volume confirmation: > 2.0x 20-period average (volume spike)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align 1d indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if missing data
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0, Bear Power < 0, strong trend (ADX > 25), volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_aligned[i] > 25 and
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bull Power < 0, Bear Power > 0, strong trend (ADX > 25), volume spike
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  adx_aligned[i] > 25 and
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (weakening bullish momentum) or ADX < 20 (trend weakening)
            if bull_power[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 (weakening bearish momentum) or ADX < 20 (trend weakening)
            if bear_power[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals