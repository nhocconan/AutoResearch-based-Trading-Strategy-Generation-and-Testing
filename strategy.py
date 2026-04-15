#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d ADX trend filter and volume spike confirmation
# Long when Williams %R(14) < -80 (oversold) + 1d ADX > 25 (strong trend) + volume > 2.0x 20-period avg
# Short when Williams %R(14) > -20 (overbought) + 1d ADX > 25 (strong trend) + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 15-25 trades/year.
# Williams %R identifies extreme reversals; ADX ensures we trade with momentum, not against it.
# Volume spike confirms institutional participation. Works in bull/bear by fading extremes in strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Williams %R (14-period) ===
    period_wr = 14
    highest_high = pd.Series(high).rolling(window=period_wr, min_periods=period_wr).max().values
    lowest_low = pd.Series(low).rolling(window=period_wr, min_periods=period_wr).min().values
    
    # Avoid division by zero
    wr = np.where((highest_high - lowest_low) != 0, 
                  -100 * (highest_high - close) / (highest_high - lowest_low), 
                  -50)  # neutral when range=0
    
    # === 1d Indicator: ADX (trend strength filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components: +DM, -DM, TR
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    
    plus_dm = np.where((high_1d - high_1d_shift) > (low_1d_shift - low_1d), 
                       np.maximum(high_1d - high_1d_shift, 0), 0)
    minus_dm = np.where((low_1d_shift - low_1d) > (high_1d - high_1d_shift), 
                        np.maximum(low_1d_shift - low_1d, 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    
    atr = np.zeros_like(tr)
    atr[period_adx-1] = np.mean(tr[:period_adx])
    for i in range(period_adx, len(tr)):
        atr[i] = (atr[i-1] * (period_adx-1) + tr[i]) / period_adx
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period_adx-1] = np.mean(plus_dm[:period_adx])
    minus_dm_smooth[period_adx-1] = np.mean(minus_dm[:period_adx])
    
    for i in range(period_adx, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period_adx-1) + plus_dm[i]) / period_adx
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period_adx-1) + minus_dm[i]) / period_adx
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Wilder's smoothing for ADX
    adx = np.zeros_like(dx)
    adx[2*period_adx-1] = np.mean(dx[period_adx-1:2*period_adx])
    for i in range(2*period_adx, len(dx)):
        adx[i] = (adx[i-1] * (period_adx-1) + dx[i]) / period_adx
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R indicates oversold (< -80)
        # 2. Strong trend (ADX > 25)
        # 3. Volume confirmation
        if (wr[i] < -80) and \
           (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R indicates overbought (> -20)
        # 2. Strong trend (ADX > 25)
        # 3. Volume confirmation
        elif (wr[i] > -20) and \
             (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR_ADX25_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0