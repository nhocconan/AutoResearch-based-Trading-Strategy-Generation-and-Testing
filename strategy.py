#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d ADX trend filter and volume confirmation
# In strong trends (ADX>25), BB breakouts have higher continuation probability.
# Volume spike confirms institutional participation. Works in bull/bear by following 1d trend.
# Target: 12-35 trades/year (50-150 total over 4 years).

name = "6h_BB_Breakout_1dADX25_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d for trend filter
    # ADX requires +DM, -DM, and TR
    dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    tr = np.maximum(
        df_1d['high'] - df_1d['low'],
        np.maximum(
            np.abs(df_1d['high'] - df_1d['close'].shift(1)),
            np.abs(df_1d['low'] - df_1d['close'].shift(1))
        )
    )
    # Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(series, period):
        result = np.zeros_like(series)
        result[period-1] = np.nansum(series[:period])
        for i in range(period, len(series)):
            result[i] = result[i-1] - (result[i-1] / period) + series[i]
        return result
    tr_14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.zeros_like(dx)
    adx[13] = np.nanmean(dx[13:27])  # First ADX value
    for i in range(27, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    adx_14_1d = adx
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate Bollinger Bands(20,2) on 6h
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for BB and ADX)
    start_idx = 34  # max(20, 20, 34) for BB, vol, ADX
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > BB upper + 1d uptrend (ADX>25 and +DI > -DI) + volume spike
            if close[i] > bb_upper[i] and adx_14_1d_aligned[i] > 25:
                # Need +DI and -DI for trend direction
                high_1d = df_1d['high'].values
                low_1d = df_1d['low'].values
                close_1d = df_1d['close'].values
                # Recalculate DI arrays for alignment (simplified: use close vs EMA)
                ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
                ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
                if close[i] > ema_10_1d_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            # Short entry: close < BB lower + 1d downtrend (ADX>25 and -DI > +DI) + volume spike
            elif close[i] < bb_lower[i] and adx_14_1d_aligned[i] > 25:
                ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
                ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
                if close[i] < ema_10_1d_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: close < BB middle or trend weakening (ADX<20)
            if close[i] < bb_ma[i] or adx_14_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: close > BB middle or trend weakening (ADX<20)
            if close[i] > bb_ma[i] or adx_14_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals