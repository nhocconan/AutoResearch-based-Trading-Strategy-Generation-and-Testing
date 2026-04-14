#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour ADX(14) and 1-day RSI(14) for trend regime filtering,
# combined with 1-hour Bollinger Band breakouts. The 4h ADX distinguishes trending (ADX>25) from ranging markets,
# while the 1d RSI avoids extremes (overbought/oversold) to prevent counter-trend entries.
# In trending markets (ADX>25), we break in the direction of the Bollinger Bands.
# In ranging markets (ADX<=25), we mean-revert at Bollinger Band extremes.
# Volume > 1.3x 20-period average confirms momentum. This avoids whipsaws in low-volatility periods.
# Position size: 0.20 (20%) to limit drawdown. Target: ~25-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Load 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # --- 4h ADX(14) for trend strength ---
    adx_len = 14
    if len(df_4h) < adx_len * 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.full(1, np.nan), tr])  # align with index
    
    # Directional Movement
    up_move = np.concatenate([np.full(1, np.nan), high_4h[1:] - high_4h[:-1]])
    down_move = np.concatenate([np.full(1, np.nan), low_4h[:-1] - low_4h[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(arr[1:period])  # skip index 0 (nan)
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, adx_len)
    plus_dm_smooth = smooth_wilder(plus_dm, adx_len)
    minus_dm_smooth = smooth_wilder(minus_dm, adx_len)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, adx_len)  # smoothed DX
    
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # --- 1d RSI(14) for regime filter ---
    rsi_len = 14
    if len(df_1d) < rsi_len:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.concatenate([np.full(1, np.nan), close_1d[1:] - close_1d[:-1]])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder smoothing for RSI
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= rsi_len:
        avg_gain[rsi_len-1] = np.nansum(gain[1:rsi_len])
        avg_loss[rsi_len-1] = np.nansum(loss[1:rsi_len])
        for i in range(rsi_len, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_len-1) + gain[i]) / rsi_len
            avg_loss[i] = (avg_loss[i-1] * (rsi_len-1) + loss[i]) / rsi_len
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- 1h Bollinger Bands (20, 2.0) ---
    bb_len = 20
    bb_std = 2.0
    if n < bb_len:
        return np.zeros(n)
    
    ma = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).mean().values
    std = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).std().values
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for all indicators
    start = max(adx_len*2, rsi_len, bb_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(ma[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        rsi_not_extreme = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter based on regime
            if trending and rsi_not_extreme and volume_confirmed:
                # Trending: break in direction of momentum
                if close[i] > upper[i]:
                    position = 1
                    signals[i] = position_size
                elif close[i] < lower[i]:
                    position = -1
                    signals[i] = -position_size
            elif not trending and volume_confirmed:
                # Ranging: mean revert at extremes
                if close[i] < lower[i]:
                    position = 1
                    signals[i] = position_size
                elif close[i] > upper[i]:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: price returns to middle band or reverses
            if close[i] > ma[i] or close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle band or reverses
            if close[i] < ma[i] or close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_ADX_1d_RSI_BB_Regime_v1"
timeframe = "1h"
leverage = 1.0