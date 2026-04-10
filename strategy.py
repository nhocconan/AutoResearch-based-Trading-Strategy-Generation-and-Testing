#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX > 25
# - Long when price breaks above 4h Donchian upper band AND 1d volume > 1.5x 20-period volume SMA AND 1d ADX > 25
# - Short when price breaks below 4h Donchian lower band AND 1d volume > 1.5x 20-period volume SMA AND 1d ADX > 25
# - Exit: price returns to 4h Donchian mid-band
# - Uses 4h for price action (Donchian channels), 1d for volume and ADX confirmation
# - Donchian breakouts capture strong momentum; volume confirms validity; ADX filters weak markets
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Tight entries target 15-30 trades/year to minimize fee drag while maintaining edge
# - Works in bull (breakouts up) and bear (breakouts down) with volume and trend filters

name = "1h_4h_1d_donchian_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return signals
    
    # Pre-compute Donchian channels on 4h (primary timeframe)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    upper_band = highest_high
    lower_band = lowest_low
    mid_band = (upper_band + lower_band) / 2.0
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing function
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1d ADX > 25 indicates sufficient trend strength
        trend_filter = adx_1d_aligned[i] > 25.0
        
        # Only trade when both volume confirmation and trend filter are present
        if vol_confirm and trend_filter:
            # Long breakout: price breaks above 4h Donchian upper band
            if close[i] > upper_band[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.20
                else:
                    signals[i] = 0.20  # Maintain position
            # Short breakout: price breaks below 4h Donchian lower band
            elif close[i] < lower_band[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.20
                else:
                    signals[i] = -0.20  # Maintain position
            # Exit: price returns to mid-band (within 0.5% of band width)
            elif abs(close[i] - mid_band[i]) < (upper_band[i] - lower_band[i]) * 0.005:
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            else:
                # Maintain current position
                signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals