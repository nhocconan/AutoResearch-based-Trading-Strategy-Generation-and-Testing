#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 4-hour Donchian breakout with weekly volume confirmation and ADX trend filter.
# Uses daily timeframe to capture medium-term trends with low transaction costs.
# Donchian breakouts capture breakouts in trending markets, while volume and ADX filters
# reduce false signals in ranging conditions. Designed for low trade frequency (<25/year)
# to minimize fee drag and improve generalization across bull/bear markets.

name = "1d_donchian20_1w_adx_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Weekly ADX(14) for trend strength
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and Directional Movement
    tr = np.full(len(close_1w), np.nan)
    dm_plus = np.full(len(close_1w), np.nan)
    dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) > 1:
        tr[0] = high_1w[0] - low_1w[0]
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_1w)):
            tr[i] = max(high_1w[i] - low_1w[i],
                       abs(high_1w[i] - close_1w[i-1]),
                       abs(low_1w[i] - close_1w[i-1]))
            dm_plus[i] = max(high_1w[i] - high_1w[i-1], 0)
            dm_minus[i] = max(low_1w[i-1] - low_1w[i], 0)
            # Wilder smoothing: only keep the larger of +DM or -DM
            if dm_plus[i] > dm_minus[i]:
                dm_minus[i] = 0
            else:
                dm_plus[i] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_1w = np.full(len(close_1w), np.nan)
    s_dm_plus = np.full(len(close_1w), np.nan)
    s_dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 14:
        atr_1w[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(close_1w)):
            atr_1w[i] = atr_1w[i-1] - (atr_1w[i-1]/14) + tr[i]
            s_dm_plus[i] = s_dm_plus[i-1] - (s_dm_plus[i-1]/14) + dm_plus[i]
            s_dm_minus[i] = s_dm_minus[i-1] - (s_dm_minus[i-1]/14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(close_1w), np.nan)
    di_minus = np.full(len(close_1w), np.nan)
    dx = np.full(len(close_1w), np.nan)
    
    for i in range(13, len(close_1w)):
        if atr_1w[i] != 0:
            di_plus[i] = 100 * s_dm_plus[i] / atr_1w[i]
            di_minus[i] = 100 * s_dm_minus[i] / atr_1w[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX calculation (smoothed DX)
    adx = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 27:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx[13:]  # Skip first 14 where DX is NaN
        if len(dx_valid) >= 14:
            adx[26] = np.nanmean(dx_valid[:14])  # First ADX at index 26
            for i in range(27, len(close_1w)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Weekly volume average (20-period) for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(19, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of Donchian 19, ADX 26, volume 19)
    start = max(19, 26, 19)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # ADX filter: only trade when trending (ADX > 25)
        trending_market = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Donchian breakdown or stoploss
            if (close[i] < donchian_low[i] or 
                close[i] < entry_price - 2.5 * (donchian_high[i] - donchian_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Donchian breakout or stoploss
            if (close[i] > donchian_high[i] or 
                close[i] > entry_price + 2.5 * (donchian_high[i] - donchian_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakouts in trending markets with volume confirmation
            if trending_market and volume_filter:
                # Long: breakout above Donchian high
                if close[i] > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low
                elif close[i] < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals