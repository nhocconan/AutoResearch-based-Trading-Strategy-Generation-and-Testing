#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets
# 1d ADX > 25 ensures we only trade when there is a strong trend to follow
# Volume spike (>1.5 x 20-period EMA) confirms breakout validity
# Works in bull markets (breakout above upper band + ADX up) and bear markets (breakdown below lower band + ADX up)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_BollingerSqueeze_Breakout_1dADX_Trend_VolumeSpike"
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
    
    # 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * bb_std_dev)
    lower_band = sma_20 - (bb_std * bb_std_dev)
    
    # Bollinger Band width for squeeze detection (low volatility)
    bb_width = (upper_band - lower_band) / sma_20
    # Squeeze condition: BB width below 20-period EMA of BB width (low volatility regime)
    bb_width_ema = pd.Series(bb_width).ewm(span=20, adjust=False, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ema
    
    # 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr > 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(bb_period, 30)  # BB period and ADX warmup
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Bollinger Band breakout above upper band with squeeze, volume confirmation and strong trend
            if close[i] > upper_band[i] and squeeze_condition[i] and volume_confirmation[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bollinger Band breakout below lower band with squeeze, volume confirmation and strong trend
            elif close[i] < lower_band[i] and squeeze_condition[i] and volume_confirmation[i] and strong_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price returns to middle band (SMA20) OR squeeze breaks (volatility expands)
            if close[i] < sma_20[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price returns to middle band (SMA20) OR squeeze breaks (volatility expands)
            if close[i] > sma_20[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals