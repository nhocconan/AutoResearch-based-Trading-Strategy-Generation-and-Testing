#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_volume_regime_v1
# Uses weekly market regime (ADX) with daily Camarilla pivot levels (H3/L3) and volume confirmation.
# In trending markets (ADX>25 on weekly), trades breakouts of H3/L3 with volume confirmation.
# In ranging markets (ADX<=25), fades touches of H3/L3 with volume confirmation.
# This adapts to both bull and bear regimes by switching between trend and mean-reversion logic.
# Target: 15-25 trades/year per symbol for low friction and high edge.

name = "1d_1w_camarilla_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for regime detection (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ADX for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(atr_1w != 0, 100 * plus_dm_smooth / atr_1w, 0)
    minus_di = np.where(atr_1w != 0, 100 * minus_dm_smooth / atr_1w, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1w = wilders_smooth(dx, 14)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align weekly ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Align daily Camarilla levels to daily timeframe (already aligned by nature of daily data)
    h3_level = camarilla_h3
    l3_level = camarilla_l3
    
    # Volume confirmation: volume > 1.3 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if not enough data
        if np.isnan(adx_1w_aligned[i]) or np.isnan(h3_level[i]) or np.isnan(l3_level[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_1w_aligned[i] > 25
        
        if is_trending:
            # TREND MODE: breakout strategy
            # Long: price breaks above H3 with volume
            if close[i] > h3_level[i] and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below L3 with volume
            elif close[i] < l3_level[i] and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit: opposite breakout
            elif close[i] < l3_level[i] and position == 1:
                position = 0
                signals[i] = 0.0
            elif close[i] > h3_level[i] and position == -1:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # RANGE MODE: mean reversion at extremes
            # Long: price touches L3 from below with volume (bounce)
            if close[i] <= l3_level[i] and low[i] < l3_level[i] and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short: price touches H3 from above with volume (rejection)
            elif close[i] >= h3_level[i] and high[i] > h3_level[i] and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit: price moves back toward center (midpoint of H3-L3)
            elif close[i] > (h3_level[i] + l3_level[i]) / 2 and position == 1:
                position = 0
                signals[i] = 0.0
            elif close[i] < (h3_level[i] + l3_level[i]) / 2 and position == -1:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals