#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w ADX trend filter.
# Long: Close breaks above Camarilla R3 AND 1d volume > 2.0x 24-period MA AND 1w ADX > 25
# Short: Close breaks below Camarilla S3 AND 1d volume > 2.0x 24-period MA AND 1w ADX > 25
# Exit: Opposite Camarilla breakout (R4/S4) or 1w ADX < 20 (range) or volume drops below 1.5x MA
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide precise intraday support/resistance; 1d volume confirms breakout strength;
# 1w ADX ensures we only trade in strong trends, avoiding whipsaws in ranging markets.
# Works in bull via long signals and bear via short signals when aligned with higher timeframe trend.

name = "12h_Camarilla_R3S3_1dVolumeSpike_1wADX25"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values  # Camarilla calculation uses open
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # True Range
    tr1 = df_1w_high - df_1w_low
    tr2 = np.abs(df_1w_high - np.roll(df_1w_close, 1))
    tr3 = np.abs(df_1w_low - np.roll(df_1w_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((df_1w_high - np.roll(df_1w_high, 1)) > (np.roll(df_1w_low, 1) - df_1w_low), 
                       np.maximum(df_1w_high - np.roll(df_1w_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1w_low, 1) - df_1w_low) > (df_1w_high - np.roll(df_1w_high, 1)), 
                        np.maximum(np.roll(df_1w_low, 1) - df_1w_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Camarilla levels for 12h timeframe using previous bar's OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_open = np.roll(open_, 1)
    
    # First bar: use current values (will be filtered by min_periods later)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_open[0] = open_[0]
    
    rang = prev_high - prev_low
    
    camarilla_r3 = prev_close + (rang * 1.1 / 4)
    camarilla_s3 = prev_close - (rang * 1.1 / 4)
    camarilla_r4 = prev_close + (rang * 1.1 / 2)
    camarilla_s4 = prev_close - (rang * 1.1 / 2)
    
    # Volume regime: current 12h volume > 2.0x 24-period MA
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        adx_val = adx_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Camarilla R3 AND volume spike AND trending
            if close_val > camarilla_r3[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 AND volume spike AND trending
            elif close_val < camarilla_s3[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Camarilla S4 OR trend weakens (ADX < 20) OR volume drops
            if close_val < camarilla_s4[i] or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Camarilla R4 OR trend weakens (ADX < 20) OR volume drops
            if close_val > camarilla_r4[i] or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals