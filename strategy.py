#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1w ADX trend filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.3x 20-period volume SMA AND 1w ADX > 20
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.3x 20-period volume SMA AND 1w ADX > 20
# - Exit: price returns to Camarilla pivot point (midpoint) or opposite Camarilla level touch
# - Uses 4h for price action, 1d for volume confirmation, 1w for trend strength filter
# - Camarilla levels provide precise intraday support/resistance; volume confirms breakout validity; ADX filters weak trends
# - Works in bull (breakouts up at H3/H4) and bear (breakouts down at L3/L4) with volume and trend filters
# - Tight entry conditions target 20-40 trades/year to minimize fee drag

name = "4h_1d_1w_camarilla_volume_adx_v1"
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
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Load 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate 1w ADX(14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Pre-compute Camarilla pivot levels on 4h (primary timeframe)
    lookback = 20
    # Calculate pivot point and support/resistance levels from prior period
    # We'll use the highest high, lowest low, and close from the lookback period
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    prev_close = pd.Series(close).shift(1).rolling(window=lookback, min_periods=lookback).last().values
    
    # Camarilla equations
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H4 = Close + Range * 1.1 / 2
    # H3 = Close + Range * 1.1 / 4
    # H2 = Close + Range * 1.1 / 6
    # H1 = Close + Range * 1.1 / 12
    # L1 = Close - Range * 1.1 / 12
    # L2 = Close - Range * 1.1 / 6
    # L3 = Close - Range * 1.1 / 4
    # L4 = Close - Range * 1.1 / 2
    
    pivot = (highest_high + lowest_low + prev_close) / 3.0
    range_val = highest_high - lowest_low
    
    h4 = pivot + range_val * 1.1 / 2
    h3 = pivot + range_val * 1.1 / 4
    l3 = pivot - range_val * 1.1 / 4
    l4 = pivot - range_val * 1.1 / 2
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w ADX > 20 indicates strong trend
        trend_filter = adx_1w_aligned[i] > 20.0
        
        # Only trade when both volume confirmation and trend filter are present
        if vol_confirm and trend_filter:
            # Long breakout: price breaks above Camarilla H3 level
            if close[i] > h3[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below Camarilla L3 level
            elif close[i] < l3[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit: price returns to pivot point (within 0.5% of range)
            elif abs(close[i] - pivot[i]) < range_val[i] * 0.005:
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            # Alternative exit: price touches opposite Camarilla level
            elif (position == 1 and close[i] < l3[i]) or (position == -1 and close[i] > h3[i]):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals