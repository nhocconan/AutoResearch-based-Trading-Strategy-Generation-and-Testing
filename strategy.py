#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX trend filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x 4h volume average AND 1d ADX > 25.
# Short when price breaks below Donchian(20) low AND volume > 1.5x 4h volume average AND 1d ADX > 25.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian and volume calculated on 4h timeframe, ADX on 1d for trend strength.
# Works in bull (breakouts with trend) and bear (breakdowns with trend).
# Target: 20-50 trades/year on 4h timeframe.

name = "4h_Donchian20_VolumeSpike_1dADX_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) on 4h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_first_1d = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr_first_1d], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # First values for DM
    dm_plus_first = np.maximum(high_1d[0] - high_1d[-1] if len(high_1d) > 1 else 0, 0) if len(high_1d) > 1 else 0
    dm_minus_first = np.maximum(low_1d[-1] - low_1d[0] if len(low_1d) > 1 else 0, 0) if len(low_1d) > 1 else 0
    dm_plus = np.concatenate([[dm_plus_first], dm_plus])
    dm_minus = np.concatenate([[dm_minus_first], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        result[period-1] = np.nansum(source[:period])
        for i in range(period, len(source)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + source[i]
            else:
                result[i] = np.nan
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, (dm_plus_smoothed / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d != 0, (dm_minus_smoothed / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Donchian, volume, and ADX
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 1.5x 4h volume average
        if vol_ma[i] <= 0 or np.isnan(vol_ma[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma[i] * 1.5)
        
        # Breakout conditions
        breakout_high = curr_close > donchian_high[i]
        breakout_low = curr_close < donchian_low[i]
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian high AND volume spike AND strong trend
            if breakout_high and volume_spike and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Breakout below Donchian low AND volume spike AND strong trend
            elif breakout_low and volume_spike and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low (failed breakout) OR ADX weakens
            elif curr_close < donchian_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high (failed breakdown) OR ADX weakens
            elif curr_close > donchian_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals