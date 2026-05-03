#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Uses 4h and 1d HTF for signal direction (EMA50 and ADX regime), 1h only for entry timing precision.
# Long when price breaks above H3 in 4h uptrend AND 1d non-bear regime (ADX<25 or price>EMA50).
# Short when price breaks below L3 in 4h downtrend AND 1d non-bull regime.
# Volume must be > 2.0x 20-period MA to confirm breakout strength.
# Uses discrete sizing 0.20 to minimize fee churn. Target: 60-150 total trades over 4 years.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# This strategy focuses on BTC and ETH as primary targets, using multi-timeframe alignment
# to avoid counter-trend trades and reduce false breakouts.

name = "1h_Camarilla_H3L3_4hEMA50_1dADX25_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial values
    tr_sum[tr_period-1] = np.sum(tr[:tr_period])
    dm_plus_sum[tr_period-1] = np.sum(dm_plus[:tr_period])
    dm_minus_sum[tr_period-1] = np.sum(dm_minus[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = np.zeros_like(di_plus)
    dx[tr_sum != 0] = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    adx_period = 14
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 2 * adx_period - 1:
        adx[2*adx_period-2] = np.mean(dx[adx_period-1:2*adx_period-1])
        for i in range(2*adx_period-1, len(dx)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d EMA50 for additional regime filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Camarilla levels from 4h OHLC (using previous bar's close)
    close_4h_shifted = np.roll(close_4h, 1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    range_4h = high_4h - low_4h
    h3 = close_4h_shifted + range_4h * 1.1 / 4
    l3 = close_4h_shifted - range_4h * 1.1 / 4
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Volume confirmation: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        # 4h trend
        trend_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        # 1d regime: non-bear for long, non-bull for short
        reg_bull = (adx_aligned[i] < 25) or (close_val > ema_50_1d_aligned[i])  # Not strong bear
        reg_bear = (adx_aligned[i] < 25) or (close_val < ema_50_1d_aligned[i])  # Not strong bull
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above H3 AND 4h uptrend AND 1d non-bear regime AND volume spike
            if close_val > h3_aligned[i] and trend_up and reg_bull and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3 AND 4h downtrend AND 1d non-bull regime AND volume spike
            elif close_val < l3_aligned[i] and trend_down and reg_bear and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR 4h trend turns down OR strong bear regime emerges
            if (close_val < l3_aligned[i] or not trend_up or 
                (adx_aligned[i] >= 25 and close_val < ema_50_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 OR 4h trend turns up OR strong bull regime emerges
            if (close_val > h3_aligned[i] or not trend_down or 
                (adx_aligned[i] >= 25 and close_val > ema_50_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals