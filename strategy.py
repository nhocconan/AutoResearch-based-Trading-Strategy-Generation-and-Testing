#!/usr/bin/env python3
# 4h_MultiTimeframe_VolatilityBreakout
# Hypothesis: Multi-timeframe volatility breakout with volume confirmation and regime filter.
# Uses 4h as primary timeframe, with 12h for trend direction and 1d for volatility regime.
# Volatility breakout triggers when price breaks beyond ATR-based bands with volume surge.
# Regime filter uses 12h ADX to avoid false signals in low-volatility environments.
# Designed to work in both bull and bear markets by adapting to volatility regimes.
# Target: 20-50 trades/year to minimize fee drag while capturing significant moves.

name = "4h_MultiTimeframe_VolatilityBreakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility measurement
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i < period:
                if i == 0:
                    atr[i] = tr[i]
                else:
                    atr[i] = (atr[i-1] * (i) + tr[i]) / (i + 1)
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    # ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            result[period-1] = np.nanmean(data[:period])
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                else:
                    result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    # Get multi-timeframe data
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_12h) < 20 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate indicators on respective timeframes
    atr_4h = calculate_atr(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volatility regime: use 1d ATR ratio (current vs 20-day average)
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = atr_1d / atr_ma_1d  # >1 = high volatility regime
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (volume_ma * 2.0)
    
    # Dynamic breakout bands based on ATR
    upper_band = df_4h['close'].values + (atr_4h * 1.5)
    lower_band = df_4h['close'].values - (atr_4h * 1.5)
    upper_band_aligned = align_htf_to_ltf(prices, df_4h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_4h, lower_band)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters: trade only in strong trend (ADX > 25) AND high volatility (vol_ratio > 1.2)
        strong_trend = adx_12h_aligned[i] > 25
        high_vol = vol_regime_aligned[i] > 1.2
        
        if position == 0:
            # Long: price breaks above upper band with volume surge and regime filters
            if (close[i] > upper_band_aligned[i] and 
                volume_surge[i] and 
                strong_trend and 
                high_vol):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume surge and regime filters
            elif (close[i] < lower_band_aligned[i] and 
                  volume_surge[i] and 
                  strong_trend and 
                  high_vol):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower band or regime deteriorates
            if (close[i] < lower_band_aligned[i]) or (adx_12h_aligned[i] < 20) or (vol_regime_aligned[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper band or regime deteriorates
            if (close[i] > upper_band_aligned[i]) or (adx_12h_aligned[i] < 20) or (vol_regime_aligned[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals