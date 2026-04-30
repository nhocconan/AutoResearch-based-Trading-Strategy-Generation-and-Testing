#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h Volume Spike and 1d ADX Trend Filter
# Bollinger Band squeeze (BBWidth < 20th percentile) indicates low volatility primed for breakout.
# Breakout direction confirmed by 12h volume spike (>2.0x 20-period average) and 1d ADX > 25.
# Uses discrete sizing 0.25 to minimize fee churn. Works in bull via upside breakouts with uptrend,
# in bear via downside breakouts with downtrend. Target: 12-37 trades/year (50-150 total over 4 years).

name = "6h_BollingerSqueeze_Breakout_12hVolume_1dADX_v1"
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
    
    # Calculate 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_ma = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std * bb_std_dev)
    bb_lower = bb_ma - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Squeeze: BBWidth < 20th percentile of lookback (50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Calculate 12h volume for confirmation (spike > 2.0x 20-period average)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h * 2.0)  # threshold array
    volume_spike = vol_12h > vol_spike_12h  # aligned inside align_htf_to_ltf? No - fix below
    
    # Recompute volume spike correctly: compare current 12h volume to its 20-period MA
    vol_ma_20_12h_values = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h_bool = vol_12h > (2.0 * vol_ma_20_12h_values)
    volume_spike = align_htf_to_ltf(prices, df_12h, vol_spike_12h_bool.astype(float)) > 0.5
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(atr == 0, 1, atr)
    di_minus = 100 * dm_minus_smooth / np.where(atr == 0, 1, atr)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    adx_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 20, 14)  # warmup for BB, percentile, vol MA, ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bb_ma[i]) or np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(volume_spike[i]) or np.isnan(adx_aligned[i])):
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
        curr_volume_spike = volume_spike[i]
        curr_adx_filter = adx_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Require BB squeeze release (end of low volatility) + volume spike + ADX trend
            if not curr_bb_squeeze and curr_volume_spike and curr_adx_filter:
                # Bullish breakout: close above upper BB
                if curr_close > curr_bb_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: close below lower BB
                elif curr_close < curr_bb_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when close returns to middle BB (mean reversion) or squeeze re-occurs
            if curr_close <= bb_ma[i] or bb_squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when close returns to middle BB or squeeze re-occurs
            if curr_close >= bb_ma[i] or bb_squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals