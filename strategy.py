#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w ADX regime filter
    # Long: Close breaks above H3 AND 1d volume > 1.5x 20-period avg AND 1w ADX > 25 (trending)
    # Short: Close breaks below L3 AND 1d volume > 1.5x 20-period avg AND 1w ADX > 25 (trending)
    # Exit: Close crosses back to H4/L4 respectively OR volume dry-up
    # Using 4h primary timeframe for optimal trade frequency (target: 75-200 total trades over 4 years)
    # Camarilla pivots from daily data provide institutional support/resistance levels
    # Volume confirmation ensures institutional participation
    # 1w ADX filter avoids choppy markets where breakouts fail
    # Discrete position sizing (0.25) to minimize fee churn
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla formulas:
    # H4 = close + 1.1 * (high - low) * 1.1/2
    # H3 = close + 1.1 * (high - low) * 1.1/4
    # H2 = close + 1.1 * (high - low) * 1.1/6
    # H1 = close + 1.1 * (high - low) * 1.1/12
    # L1 = close - 1.1 * (high - low) * 1.1/12
    # L2 = close - 1.1 * (high - low) * 1.1/6
    # L3 = close - 1.1 * (high - low) * 1.1/4
    # L4 = close - 1.1 * (high - low) * 1.1/2
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_range = prev_high - prev_low
    h4 = prev_close + 1.1 * camarilla_range * 1.1 / 2
    h3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    h2 = prev_close + 1.1 * camarilla_range * 1.1 / 6
    h1 = prev_close + 1.1 * camarilla_range * 1.1 / 12
    l1 = prev_close - 1.1 * camarilla_range * 1.1 / 12
    l2 = prev_close - 1.1 * camarilla_range * 1.1 / 6
    l3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    l4 = prev_close - 1.1 * camarilla_range * 1.1 / 2
    
    # Align daily Camarilla levels to 4h
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    h2_4h = align_htf_to_ltf(prices, df_1d, h2)
    h1_4h = align_htf_to_ltf(prices, df_1d, h1)
    l1_4h = align_htf_to_ltf(prices, df_1d, l1)
    l2_4h = align_htf_to_ltf(prices, df_1d, l2)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate weekly ADX for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing for ADX
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_wild = wilders_smoothing(tr, period)
    dm_plus_wild = wilders_smoothing(dm_plus, period)
    dm_minus_wild = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = np.where(atr_wild != 0, 100 * dm_plus_wild / atr_wild, 0)
    di_minus = np.where(atr_wild != 0, 100 * dm_minus_wild / atr_wild, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align weekly ADX to 4h
    adx_4h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate daily volume MA for confirmation (>1.5x 20-period average)
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(df_1d['volume'].values[i-20:i])
    
    volume_spike_1d = df_1d['volume'].values > (1.5 * vol_ma_1d)
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(adx_4h[i]) or np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market (good for breakouts)
        trending_regime = adx_4h[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike_4h[i]
        
        # Entry logic: Camarilla breakout + volume confirmation + trending regime
        long_entry = (close[i] > h3_4h[i]) and vol_confirm and trending_regime
        short_entry = (close[i] < l3_4h[i]) and vol_confirm and trending_regime
        
        # Exit logic: Close crosses back to H4/L4 respectively OR volume dry-up
        long_exit = (close[i] < h4_4h[i]) or not vol_confirm
        short_exit = (close[i] > l4_4h[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_camarilla_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0