#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Bollinger Band width percentile identifies low volatility squeezes (regime filter).
# Breakout from squeezed BB with volume spike and 1d ADX > 25 (trending regime) captures explosive moves.
# Works in bull via upside breakouts, in bear via downside breakouts. ADX filter avoids whipsaws in ranging markets.
# Discrete sizing 0.25 balances risk and minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_BB_Squeeze_Breakout_1dADX_VolumeSpike_v1"
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
    
    # Calculate 1d Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    bb_ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma_20 + (2.0 * bb_std_20)
    bb_lower = bb_ma_20 - (2.0 * bb_std_20)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (50-period lookback) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze_condition = bb_width_percentile < 20  # Bottom 20% = squeeze
    
    # Align 1d BB width percentile to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    
    # Calculate 1d ADX(14) for trend filter
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
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
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / atr)
    di_minus = 100 * (dm_minus_smooth / atr)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for BB percentile and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_squeeze = squeeze_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Bollinger Bands on 6h timeframe for breakout detection
        bb_ma_20_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().iloc[i]
        bb_std_20_6h = pd.Series(close).rolling(window=20, min_periods=20).std().iloc[i]
        bb_upper_6h = bb_ma_20_6h + (2.0 * bb_std_20_6h)
        bb_lower_6h = bb_ma_20_6h - (2.0 * bb_std_20_6h)
        
        if position == 0:  # Flat - look for new entries
            # Require squeeze breakout, volume spike, and trending regime (ADX > 25)
            if curr_squeeze and curr_volume_spike and curr_adx > 25:
                # Bullish breakout: close above upper BB
                if curr_close > bb_upper_6h:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: close below lower BB
                elif curr_close < bb_lower_6h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price re-enters Bollinger Bands (mean reversion) or ADX weakens
            if curr_close < bb_ma_20_6h or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price re-enters Bollinger Bands (mean reversion) or ADX weakens
            if curr_close > bb_ma_20_6h or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals