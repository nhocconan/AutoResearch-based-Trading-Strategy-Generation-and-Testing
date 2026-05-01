#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX regime filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d ADX > 25 filters for trending markets (avoid chop) while ADX < 20 identifies range-bound markets
# In trending markets (ADX > 25): Williams %R < -80 for long, > -20 for short (pullback entries)
# In ranging markets (ADX < 20): Williams %R < -90 for long, > -10 for short (extreme reversals)
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~12-37 trades/year per symbol with 0.25 sizing
# Works in both bull and bear markets by adapting to regime conditions

name = "6h_WilliamsR_1dADX_Regime_Volume_v1"
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
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])  # skip first NaN
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R (14-period) on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d ADX (30 days) + Williams %R (14 periods) + volume EMA20
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(wr[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        williams_r_val = wr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Regime-based entry logic
            if adx > 25:  # Trending market
                # Look for pullbacks in trend direction
                if close[i] > close[i-1]:  # Uptrend bias (price rising)
                    if williams_r_val < -80 and vol_spike:  # Oversold pullback
                        signals[i] = 0.25
                        position = 1
                    else:
                        signals[i] = 0.0
                elif close[i] < close[i-1]:  # Downtrend bias (price falling)
                    if williams_r_val > -20 and vol_spike:  # Overbought pullback
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0  # No clear trend bias
            elif adx < 20:  # Ranging market
                # Look for extreme reversals
                if williams_r_val < -90 and vol_spike:  # Extremely oversold
                    signals[i] = 0.25
                    position = 1
                elif williams_r_val > -10 and vol_spike:  # Extremely overbought
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # Transition regime (20 <= ADX <= 25)
                signals[i] = 0.0  # Avoid choppy transition
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (momentum loss) or ADX weakening
            if williams_r_val > -50 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (momentum loss) or ADX weakening
            if williams_r_val < -50 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals