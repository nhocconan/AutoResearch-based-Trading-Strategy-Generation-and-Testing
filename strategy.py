#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily ADX trend filter and volume spike confirmation
# on 12h price breaks of daily Donchian channels (20-period). This combines trend
# following with volatility breakouts, targeting 15-25 trades/year per symbol.
# Works in bull markets (trend continuation) and bear markets (mean reversion via
# volatility expansion after panic). Uses discrete position sizing (0.25) to limit
# fee churn. ADX > 25 ensures we only trade in trending regimes, reducing false
# breakouts in ranging markets.

name = "12h_1d_ADX_Trend_DonchianBreakout_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Donchian Channels (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper and lower bands (20-period high/low)
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # === Daily ADX (14-period) for trend strength ===
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR and DM
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    atr[tr_period-1] = np.mean(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.mean(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.mean(dm_minus[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / np.where(atr != 0, atr, 1e-10)
    minus_di = 100 * dm_minus_smooth / np.where(atr != 0, atr, 1e-10)
    
    # DX and ADX
    dx = np.zeros_like(plus_di)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = np.mean(dx[:2*tr_period])  # seed
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for indicators
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = upper_band_aligned[i]
        lower_val = lower_band_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(adx_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian band with volume confirmation in trending market
            if close_val > upper_val and vol_ratio_val > 2.0 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian band with volume confirmation in trending market
            elif close_val < lower_val and vol_ratio_val > 2.0 and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below lower Donchian band OR trend weakens
            if close_val < lower_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above upper Donchian band OR trend weakens
            if close_val > upper_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals