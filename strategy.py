#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX trend filter and volume confirmation
# Bollinger Squeeze (BB Width < 20th percentile) identifies low volatility primed for expansion
# Breakout direction confirmed by 1d ADX > 25 (trending) and volume spike > 2.0x 20-period average
# Works in bull markets via upward breakouts with ADX confirmation
# Works in bear markets via downward breakouts with ADX confirmation
# 6h timeframe targets 12-37 trades/year to minimize fee drag while capturing explosive moves

name = "6h_BB_Squeeze_ADX_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period]) if np.any(~np.isnan(data[1:period])) else 0
        # Rest is Wilder smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = WilderSmooth(tr, period)
    plus_dm_smooth = WilderSmooth(plus_dm, period)
    minus_dm_smooth = WilderSmooth(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmooth(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std_dev * bb_std)
    bb_lower = bb_ma - (bb_std_dev * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Squeeze: BB Width < 20th percentile of lookback (50 periods)
    lookback = 50
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(lookback, len(bb_width)):
        window = bb_width[i-lookback:i]
        if not np.all(np.isnan(window)):
            bb_width_percentile[i] = np.nanpercentile(window, 20)
    
    bb_squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(30 + 14 + 14, bb_period, 20, lookback)  # ADX + BB + volume
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bb_ma[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Bollinger Squeeze condition
        squeeze = bb_squeeze[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze breakout above upper band, trending, volume spike
            if squeeze and close[i] > bb_upper[i] and trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower band, trending, volume spike
            elif squeeze and close[i] < bb_lower[i] and trending and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below middle band or loss of trend/volume
            if close[i] < bb_ma[i] or not trending or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above middle band or loss of trend/volume
            if close[i] > bb_ma[i] or not trending or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals