#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1w ADX trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets.
# 1w ADX > 25 filters for trending regimes to avoid false breakouts in chop.
# Volume spike confirms breakout validity. Works in bull via upside breakouts, in bear via downside breakouts.
# Discrete sizing 0.25 balances risk and minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_BB_Squeeze_Breakout_1wADX_VolumeSpike_v1"
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
    
    # Calculate 6h Bollinger Bands (20, 2.0)
    close_s = pd.Series(close)
    bb_ma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std_20 = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma_20 + (2.0 * bb_std_20)
    bb_lower = bb_ma_20 - (2.0 * bb_std_20)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Squeeze: width < 20-period average width (low volatility)
    bb_width_ma_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma_20
    
    # Calculate 1w ADX(14) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # first bar
    tr2[0] = np.abs(high_1w[0] - close_1w[0])  # first bar
    tr3[0] = np.abs(low_1w[0] - close_1w[0])  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_14 / tr_14)
    di_minus = 100 * (dm_minus_14 / tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1w ADX to 6h timeframe (wait for completed 1w bar)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # warmup for BB and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(bb_width_ma_20[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_squeeze = bb_squeeze[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require Bollinger Band squeeze, ADX > 25 (trending regime), and volume spike
            if curr_bb_squeeze and (curr_adx > 25) and curr_volume_spike:
                # Bullish breakout: price breaks above upper BB
                if curr_close > curr_bb_upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lower BB
                elif curr_close < curr_bb_lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price returns to middle BB (mean reversion) or breaks below lower BB (stop)
            if curr_close < bb_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to middle BB (mean reversion) or breaks above upper BB (stop)
            if curr_close > bb_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals