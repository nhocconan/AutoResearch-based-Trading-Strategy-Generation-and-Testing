#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h ADX > 20
# - Long when price breaks above 4h Donchian upper band AND 12h volume > 1.3x 20-period volume SMA AND 12h ADX > 20
# - Short when price breaks below 4h Donchian lower band AND 12h volume > 1.3x 20-period volume SMA AND 12h ADX > 20
# - Exit: price returns to 4h Donchian mid-band or ATR trailing stop (2.0*ATR)
# - Uses 4h for price action (Donchian channels), 12h for volume and ADX confirmation
# - Donchian breakouts capture strong momentum; volume confirms validity; ADX filters weak markets
# - Tight entries target 20-50 trades/year to minimize fee drag while maintaining edge
# - Works in bull (breakouts up) and bear (breakouts down) with volume and trend filters

name = "4h_12h_donchian_volume_adx_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Calculate 12h volume SMA for confirmation
    vol_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Calculate 12h ADX(14) for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
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
    
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute Donchian channels on 4h (primary timeframe)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    upper_band = highest_high
    lower_band = lowest_low
    mid_band = (upper_band + lower_band) / 2.0
    
    # ATR for dynamic stoploss (using 4h data)
    tr_4h1 = np.abs(high[1:] - low[:-1])
    tr_4h2 = np.abs(high[1:] - close[:-1])
    tr_4h3 = np.abs(low[1:] - close[:-1])
    tr_4h = np.maximum(np.maximum(tr_4h1, tr_4h2), tr_4h3)
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
        vol_confirm = vol_12h_aligned[i] > 1.3 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 12h ADX > 20 indicates sufficient trend strength
        trend_filter = adx_12h_aligned[i] > 20.0
        
        # Only trade when both volume confirmation and trend filter are present
        if vol_confirm and trend_filter:
            # Long breakout: price breaks above Donchian upper band
            if close[i] > upper_band[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below Donchian lower band
            elif close[i] < lower_band[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit: price returns to mid-band (within 0.1% of band width)
            elif abs(close[i] - mid_band[i]) < (upper_band[i] - lower_band[i]) * 0.001:
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            # Alternative exit: ATR-based trailing stop
            elif position == 1 and close[i] < (highest_high_since_entry - 2.0 * atr_4h[i]):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            elif position == -1 and close[i] > (lowest_low_since_entry + 2.0 * atr_4h[i]):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            else:
                # Maintain current position and update tracking levels
                if position == 1:
                    # Track highest high since entry for trailing stop
                    if 'highest_high_since_entry' not in locals():
                        highest_high_since_entry = high[i]
                    else:
                        highest_high_since_entry = max(highest_high_since_entry, high[i])
                    signals[i] = 0.25
                elif position == -1:
                    # Track lowest low since entry for trailing stop
                    if 'lowest_low_since_entry' not in locals():
                        lowest_low_since_entry = low[i]
                    else:
                        lowest_low_since_entry = min(lowest_low_since_entry, low[i])
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                # Reset tracking variables
                if 'highest_high_since_entry' in locals():
                    del highest_high_since_entry
                if 'lowest_low_since_entry' in locals():
                    del lowest_low_since_entry
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals