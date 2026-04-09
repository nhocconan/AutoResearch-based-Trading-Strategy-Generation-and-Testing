#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX + 1d volume confirmation
# Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# ADX(14) from 12h filters regime: ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = ranging (fade Elder Ray extremes)
# 1d volume spike (volume > 1.5 * 20-period average volume) confirms breakout authenticity
# Works in bull/bear: regime filter adapts, Elder Ray captures momentum in trends and mean reversion in ranges
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_12h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 12h ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr_12h, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h > 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h > 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5 * 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending_regime = adx_aligned[i] > 25
        ranging_regime = adx_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: Elder Ray turns bearish OR regime shifts to ranging
            if bear_power[i] > 0 or ranging_regime:  # Bear power positive means bears in control
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray turns bullish OR regime shifts to ranging
            if bull_power[i] < 0 or ranging_regime:  # Bull power negative means bulls in control
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime and volume_confirmed:
                # Follow Elder Ray in trending regime
                if bull_power[i] > 0 and bear_power[i] < 0:  # Strong bullish momentum
                    position = 1
                    signals[i] = 0.25
                elif bear_power[i] > 0 and bull_power[i] < 0:  # Strong bearish momentum
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirmed:
                # Fade Elder Ray extremes in ranging regime
                if bear_power[i] > 0 and np.abs(bear_power[i]) > np.abs(bull_power[i]):  # Extreme bearish, fade to long
                    position = 1
                    signals[i] = 0.25
                elif bull_power[i] < 0 and np.abs(bull_power[i]) > np.abs(bear_power[i]):  # Extreme bullish, fade to short
                    position = -1
                    signals[i] = -0.25
    
    return signals