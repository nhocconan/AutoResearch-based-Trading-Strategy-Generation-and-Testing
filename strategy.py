#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime + Volume Spike
# Elder Ray measures bull/bear power via EMA13. 1d ADX regime filter ensures we only trade
# in strong trends (ADX>25) or range (ADX<20) with hysteresis to prevent whipsaw.
# Volume confirmation filters low-conviction moves. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via bull power > 0 in uptrend and bear markets via bear power < 0 in downtrend.

name = "6h_ElderRay_1dADXRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = df_1d['close'].shift(1)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - prev_close)
    tr3 = abs(df_1d['low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # +DM = max(high - prev_high, 0) if > prev_low - low else 0
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    up_move = df_1d['high'] - prev_high
    down_move = prev_low - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # +DI, -DI, ADX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Regime with hysteresis: ADX>25 = trending, ADX<20 = ranging
    adx_regime = np.full(len(adx_values), np.nan)
    regime = 0  # 0: undefined, 1: trending, -1: ranging
    for i in range(len(adx_values)):
        if np.isnan(adx_values[i]):
            adx_regime[i] = np.nan
        elif regime == 0:
            if adx_values[i] > 25:
                regime = 1
                adx_regime[i] = 1
            elif adx_values[i] < 20:
                regime = -1
                adx_regime[i] = -1
            else:
                adx_regime[i] = np.nan
        elif regime == 1:  # currently trending
            if adx_values[i] < 20:
                regime = -1
                adx_regime[i] = -1
            else:
                adx_regime[i] = 1
        elif regime == -1:  # currently ranging
            if adx_values[i] > 25:
                regime = 1
                adx_regime[i] = 1
            else:
                adx_regime[i] = -1
    
    # Align ADX regime to 6h timeframe
    adx_regime_aligned = align_htf_to_ltf(prices, df_1d, adx_regime)
    
    # Calculate Elder Ray on 6h: Bull Power = high - EMA13, Bear Power = low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_regime_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: bull power > 0 in trending regime (ADX>25) with volume spike
            if bull_power[i] > 0 and adx_regime_aligned[i] == 1 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bear power < 0 in trending regime (ADX>25) with volume spike
            elif bear_power[i] < 0 and adx_regime_aligned[i] == 1 and volume_spike:
                signals[i] = -0.25
                position = -1
            # Long in ranging regime: bull power > 0 (mean reversion from low)
            elif bull_power[i] > 0 and adx_regime_aligned[i] == -1 and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short in ranging regime: bear power < 0 (mean reversion from high)
            elif bear_power[i] < 0 and adx_regime_aligned[i] == -1 and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: bull power <= 0 or regime shifts against position
            if bull_power[i] <= 0 or (adx_regime_aligned[i] == 1 and bear_power[i] < 0) or (adx_regime_aligned[i] == -1 and bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power >= 0 or regime shifts against position
            if bear_power[i] >= 0 or (adx_regime_aligned[i] == 1 and bull_power[i] > 0) or (adx_regime_aligned[i] == -1 and bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals