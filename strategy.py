#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d ADX regime filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions (-20 to 0 = overbought, -80 to -100 = oversold).
# 1d ADX > 25 filters for trending markets (avoid false signals in chop), ADX < 20 for ranging markets.
# Volume confirmation (>1.8x 24-period average) ensures institutional participation.
# In trending markets (ADX>25): mean revert at extremes (short at %R>-20, long at %R<-80).
# In ranging markets (ADX<20): fade moves from 1d VWAP (short above VWAP, long below VWAP).
# Discrete sizing (0.25) minimizes fee churn. Effective in both bull and bear markets by adapting to regime.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsR_1dADX_Regime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_14 = adx
    
    # Align 1d ADX to 12h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 1d VWAP for ranging regime
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (typical_price_1d * df_1d['volume'].values).cumsum() / df_1d['volume'].values.cumsum()
    vwap_1d[np.isnan(vwap_1d)] = 0  # Handle division by zero
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Williams %R on 12h timeframe (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Calculate 24-period average volume for confirmation (on 12h timeframe)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 24)  # Williams %R, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_14_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_adx = adx_14_aligned[i]
        curr_vwap = vwap_1d_aligned[i]
        curr_vol_ma = vol_ma_24[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 24-period average
        vol_confirm = curr_volume > 1.8 * curr_vol_ma
        
        # Regime determination
        is_trending = curr_adx > 25
        is_ranging = curr_adx < 20
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions based on regime
            if is_trending:
                # In trending market: exit when %R returns from oversold
                if curr_wr > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging market
                # In ranging market: exit when price crosses VWAP
                if curr_close > curr_vwap:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions based on regime
            if is_trending:
                # In trending market: exit when %R returns from overbought
                if curr_wr < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging market
                # In ranging market: exit when price crosses VWAP
                if curr_close < curr_vwap:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
                    
        else:  # Flat - look for new entries
            if is_trending:
                # Trending regime: mean reversion at extremes
                # Long entry: Williams %R deeply oversold AND volume confirmation
                if (curr_wr < -80 and vol_confirm):
                    signals[i] = 0.25
                    position = 1
                # Short entry: Williams %R deeply overbought AND volume confirmation
                elif (curr_wr > -20 and vol_confirm):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: fade moves from VWAP
                # Long entry: price below VWAP AND volume confirmation
                if (curr_close < curr_vwap and vol_confirm):
                    signals[i] = 0.25
                    position = 1
                # Short entry: price above VWAP AND volume confirmation
                elif (curr_close > curr_vwap and vol_confirm):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX between 20-25): no trading
                signals[i] = 0.0
    
    return signals