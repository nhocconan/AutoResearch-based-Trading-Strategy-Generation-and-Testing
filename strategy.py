#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w ADX regime filter
    # Long: price breaks above H3 pivot AND volume > 1.5x 20-period avg AND 1w ADX > 25 (trending)
    # Short: price breaks below L3 pivot AND volume > 1.5x 20-period avg AND 1w ADX > 25 (trending)
    # Exit: price retreats to H4/L4 pivot levels OR volume dry-up OR 1w ADX < 20 (range)
    # Using 4h primary for balance of signal quality and trade frequency.
    # Camarilla pivots provide mathematically derived support/resistance levels.
    # Volume confirmation ensures institutional participation.
    # 1w ADX regime filter avoids whipsaws in sideways markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
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
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Camarilla formulas:
    # H4 = close + 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # H2 = close + 1.1*(high-low)*1.1/6
    # H1 = close + 1.1*(high-low)*1.1/12
    # L1 = close - 1.1*(high-low)*1.1/12
    # L2 = close - 1.1*(high-low)*1.1/6
    # L3 = close - 1.1*(high-low)*1.1/4
    # L4 = close - 1.1*(high-low)*1.1/2
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels
    camarilla_h4 = np.full(len(daily_close), np.nan)
    camarilla_h3 = np.full(len(daily_close), np.nan)
    camarilla_l3 = np.full(len(daily_close), np.nan)
    camarilla_l4 = np.full(len(daily_close), np.nan)
    
    for i in range(len(daily_close)):
        if i >= 1:  # Need previous day's data
            high_low = daily_high[i-1] - daily_low[i-1]
            camarilla_h4[i] = daily_close[i-1] + 1.1 * high_low * 1.1 / 2
            camarilla_h3[i] = daily_close[i-1] + 1.1 * high_low * 1.1 / 4
            camarilla_l3[i] = daily_close[i-1] - 1.1 * high_low * 1.1 / 4
            camarilla_l4[i] = daily_close[i-1] - 1.1 * high_low * 1.1 / 2
    
    # Align Camarilla levels to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate daily volume confirmation (>1.5x 20-period average)
    daily_volume = df_1d['volume'].values
    vol_ma = np.full(len(daily_volume), np.nan)
    for i in range(20, len(daily_volume)):
        vol_ma[i] = np.mean(daily_volume[i-20:i])
    volume_spike = daily_volume > (1.5 * vol_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate weekly ADX for regime filter
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # True Range calculation
    tr1 = np.abs(weekly_high[1:] - weekly_low[1:])
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((weekly_high[1:] - weekly_high[:-1]) > (weekly_low[:-1] - weekly_low[1:]),
                       np.maximum(weekly_high[1:] - weekly_high[:-1], 0), 0)
    dm_minus = np.where((weekly_low[:-1] - weekly_low[1:]) > (weekly_high[1:] - weekly_high[:-1]),
                        np.maximum(weekly_low[:-1] - weekly_low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market (favor breakouts)
        trending_market = adx_aligned[i] > 25
        ranging_market = adx_aligned[i] < 20  # Exit condition
        
        # Volume confirmation
        vol_confirm = volume_spike_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + regime
        long_entry = (close[i] > h3_aligned[i]) and vol_confirm and trending_market
        short_entry = (close[i] < l3_aligned[i]) and vol_confirm and trending_market
        
        # Exit logic: retreat to H4/L4 OR volume dry-up OR ranging market
        long_exit = (close[i] < h4_aligned[i]) or not vol_confirm or ranging_market
        short_exit = (close[i] > l4_aligned[i]) or not vol_confirm or ranging_market
        
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