#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear strength relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA
# Strong trends: Bull Power > 0 + Bear Power < 0 with expansion; ADX > 25 confirms trend regime
# Volume spike (2.0x 24-period average) confirms institutional participation
# Works in bull markets via sustained bull power and bear markets via bear power with ADX filter
# Discrete sizing 0.25 minimizes fee churn. Target: 75-150 total trades over 4 years (19-37/year).

name = "6h_ElderRay_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h EMA13 for Elder Ray calculation (primary timeframe)
    ema_period = 13
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Load 1d data ONCE before loop for ADX trend filter (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar has no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    adx_period = 14
    alpha = 1.0 / adx_period
    tr_smooth = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    tr_smooth[adx_period-1] = np.mean(tr[:adx_period])
    dm_plus_smooth[adx_period-1] = np.mean(dm_plus[:adx_period])
    dm_minus_smooth[adx_period-1] = np.mean(dm_minus[:adx_period])
    
    # Wilder's smoothing
    for i in range(adx_period, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / adx_period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / adx_period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / adx_period) + dm_minus[i]
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.zeros_like(di_plus)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[di_plus + di_minus == 0] = 0  # Avoid division by zero
    
    adx = np.zeros_like(dx)
    adx[2*adx_period-1] = np.mean(dx[adx_period:2*adx_period])  # First ADX value
    for i in range(2*adx_period, len(dx)):
        adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*6h = 144h = 6 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_period, 24, 2*adx_period)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require ADX > 25 (trending market) and volume spike
            if curr_adx > 25 and curr_volume_spike:
                # Bullish entry: Bull Power > 0 and Bear Power < 0 (strong bull momentum)
                if curr_bull_power > 0 and curr_bear_power < 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 and Bull Power < 0 (strong bear momentum)
                elif curr_bear_power < 0 and curr_bull_power < 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when bull power turns negative OR ADX drops below 20 (trend weakening)
            if curr_bull_power <= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bear power turns positive OR ADX drops below 20 (trend weakening)
            if curr_bear_power >= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals