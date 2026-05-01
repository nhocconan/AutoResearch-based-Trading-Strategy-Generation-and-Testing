#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
# Long when: BB squeeze (bandwidth < 20th percentile) AND price breaks above upper band AND 1d ADX > 25 AND 6h volume > 1.5x 20-period average
# Short when: BB squeeze AND price breaks below lower band AND 1d ADX > 25 AND 6h volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 12-37 trades/year on 6h.
# Bollinger squeeze identifies low volatility primed for breakout, ADX ensures breakout occurs in trending environment, volume confirms conviction.
# Works in bull (catching bullish breakouts) and bear (catching bearish breakdowns) by trading breakouts in the direction of the 1d trend.

name = "6h_BB_Squeeze_Breakout_1dADX_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Bollinger Bands
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Bollinger Bands on 6h (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma_6h = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_6h = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_6h + (bb_std * std_6h)
    lower_band = sma_6h - (bb_std * std_6h)
    bandwidth = (upper_band - lower_band) / sma_6h  # Normalized bandwidth
    
    # Al Bollinger Bands to 6h primary timeframe (already aligned as computed from close)
    # Note: sma_6h, std_6h, upper_band, lower_band, bandwidth are already 6h-aligned as they use close prices
    
    # 1d ADX for trend filter (14-period)
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
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = alpha = 1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_1d = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_1d = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus_1d = 100 * (dm_plus_1d / atr_1d)
    di_minus_1d = 100 * (dm_minus_1d / atr_1d)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h volume average (20-period) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate bandwidth percentile rank (20-period lookback) for squeeze detection
    bandwidth_series = pd.Series(bandwidth)
    bandwidth_percentile = bandwidth_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for bandwidth percentile and ADX
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(sma_6h[i]) or np.isnan(std_6h[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(bandwidth_percentile[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_bandwidth_pct = bandwidth_percentile[i]
        curr_adx = adx_aligned[i]
        
        # Bollinger squeeze: bandwidth < 20th percentile (low volatility)
        squeeze = curr_bandwidth_pct < 20.0
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 1d trend filter: ADX > 25 indicates trending market
        trending = curr_adx > 25.0
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: squeeze AND price breaks above upper band AND trending AND volume confirmation
            if (squeeze and 
                curr_close > curr_upper and 
                trending and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: squeeze AND price breaks below lower band AND trending AND volume confirmation
            elif (squeeze and 
                  curr_close < curr_lower and 
                  trending and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below middle band (SMA) OR squeeze breaks (volatility expands)
            if (curr_close < sma_6h[i] or 
                curr_bandwidth_pct > 50.0):  # Exit when bandwidth > median
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above middle band (SMA) OR squeeze breaks (volatility expands)
            if (curr_close > sma_6h[i] or 
                curr_bandwidth_pct > 50.0):  # Exit when bandwidth > median
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals