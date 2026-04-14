#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with weekly volume confirmation and weekly ADX trend filter
# Long when price breaks above weekly Donchian high + weekly volume > 1.5x avg + weekly ADX > 25
# Short when price breaks below weekly Donchian low + weekly volume > 1.5x avg + weekly ADX > 25
# Exit when price crosses the weekly Donchian midpoint
# Weekly timeframe reduces trade frequency to avoid fee drag, weekly Donchian captures longer-term trends,
# Volume confirms institutional interest, ADX ensures trending conditions to avoid whipsaws in ranging markets
# Target: 30-100 trades over 4 years (7-25/year) with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d weekly Donchian channels (20-period)
    def donchian_channels(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1d weekly volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly data for volume and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    
    # Weekly ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned weekly indicators
        vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)[i]
        adx_aligned = align_htf_to_ltf(prices, df_1w, adx)[i]
        
        # Check for NaN values
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(vol_ma_1w_aligned) or np.isnan(adx_aligned)):
            continue
        
        # Volume confirmation (> 1.5x average on both timeframes)
        volume_confirm_1d = volume[i] > 1.5 * vol_ma[i]
        volume_confirm_1w = vol_1w[i] > 1.5 * vol_ma_1w_aligned if i < len(vol_1w) else False
        volume_confirm = volume_confirm_1d and volume_confirm_1w
        
        # ADX trend filter (> 25)
        trend_filter = adx_aligned > 25
        
        if position == 0:  # No position - look for entries
            if volume_confirm and trend_filter:
                # Long: break above Donchian upper
                if close[i] > donchian_upper[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below Donchian lower
                elif close[i] < donchian_lower[i]:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price crosses below midpoint
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price crosses above midpoint
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Volume_ADX"
timeframe = "1d"
leverage = 1.0