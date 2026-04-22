#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume Weighted MACD with 1d ADX trend filter and volume confirmation
# Uses volume-weighted MACD for better momentum signal in low-volume periods
# ADX(14) from 1d filters for trending markets only (ADX > 25)
# Volume spike (2x 20-period average) confirms institutional participation
# Designed for 6h timeframe to target 15-30 trades/year per symbol.
# Volume weighting reduces false signals during low-volume consolidations
# Works in bull markets (captures momentum) and bear (avoids false signals via ADX filter)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for higher timeframe trend strength filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # First value has no previous close
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                for i in range(period, len(data)):
                    if not np.isnan(data[i]):
                        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # Avoid division by zero
        dm_plus_di = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        dm_minus_di = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        dx = np.where((dm_plus_di + dm_minus_di) != 0, 
                      100 * np.abs(dm_plus_di - dm_minus_di) / (dm_plus_di + dm_minus_di), 0)
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Volume-weighted MACD components
    # Volume-weighted close price
    vwc = (close * volume) / np.where(volume != 0, volume, 1)
    vwc[volume == 0] = close[volume == 0]  # Fallback to regular close if no volume
    
    # EMA of volume-weighted close
    def ema_wilder(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # Simple average for first value
            result[period-1] = np.mean(data[:period])
            # Wilder smoothing for subsequent values
            alpha = 2.0 / (period + 1)
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    vwc_ema12 = ema_wilder(vwc, 12)
    vwc_ema26 = ema_wilder(vwc, 26)
    macd_line = vwc_ema12 - vwc_ema26
    macd_signal = ema_wilder(macd_line, 9)
    macd_histogram = macd_line - macd_signal
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(macd_line[i]) or 
            np.isnan(macd_signal[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: MACD bullish crossover + strong trend (ADX > 25) + volume spike
            if (macd_line[i] > macd_signal[i] and 
                macd_line[i-1] <= macd_signal[i-1] and 
                adx_14_1d_aligned[i] > 25 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: MACD bearish crossover + strong trend (ADX > 25) + volume spike
            elif (macd_line[i] < macd_signal[i] and 
                  macd_line[i-1] >= macd_signal[i-1] and 
                  adx_14_1d_aligned[i] > 25 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: MACD crossover in opposite direction or trend weakening
            if position == 1:
                # Exit on MACD bearish crossover or ADX < 20 (trend weakening)
                if (macd_line[i] < macd_signal[i] and 
                    macd_line[i-1] >= macd_signal[i-1]) or \
                   adx_14_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on MACD bullish crossover or ADX < 20 (trend weakening)
                if (macd_line[i] > macd_signal[i] and 
                    macd_line[i-1] <= macd_signal[i-1]) or \
                   adx_14_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_VolumeWeightedMACD_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0