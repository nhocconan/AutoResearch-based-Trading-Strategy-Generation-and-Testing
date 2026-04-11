#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels (using previous day's data to avoid look-ahead)
    r4 = pivot + (range_12h * 1.1 / 2)
    r3 = pivot + (range_12h * 1.1 / 4)
    r2 = pivot + (range_12h * 1.1 / 6)
    r1 = pivot + (range_12h * 1.1 / 12)
    s1 = pivot - (range_12h * 1.1 / 12)
    s2 = pivot - (range_12h * 1.1 / 6)
    s3 = pivot - (range_12h * 1.1 / 4)
    s4 = pivot - (range_12h * 1.1 / 2)
    
    # Shift by 1 to use only completed 12h bars
    r4 = np.roll(r4, 1)
    r3 = np.roll(r3, 1)
    r2 = np.roll(r2, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    s2 = np.roll(s2, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    r4[0] = np.nan
    r3[0] = np.nan
    r2[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    s2[0] = np.nan
    s3[0] = np.nan
    s4[0] = np.nan
    
    # Align 12h Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: ADX > 25 for trending market
    # Calculate ADX components
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Smooth +DM and -DM
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    trending_market = adx > 25
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Entry signals - only in trending markets
        long_signal = False
        short_signal = False
        
        # Long: price breaks above R3 with volume and trend
        if price_high > r3_aligned[i] and volume_confirmed and trending:
            long_signal = True
        
        # Short: price breaks below S3 with volume and trend
        if price_low < s3_aligned[i] and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr[i])
        
        # Exit when price returns to pivot (mean reversion within trend)
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
        if np.isnan(pivot_aligned[i]):
            pivot_val = 0
        else:
            pivot_val = pivot_aligned[i]
        
        exit_long = position == 1 and price_close < pivot_val
        exit_short = position == -1 and price_close > pivot_val
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla breakout strategy with volume confirmation and ADX trend filter on 12h timeframe.
# Enters long when price breaks above R3 (Camarilla resistance level 3) with volume confirmation (>1.5x avg volume) in trending markets (ADX > 25).
# Enters short when price breaks below S3 (Camarilla support level 3) with volume confirmation and ADX > 25.
# Uses 12h timeframe for Camarilla levels to capture multi-day intraday swings.
# Volume confirmation ensures institutional participation, ADX filter avoids whipsaws in sideways markets.
# Exits when price returns to the pivot point or ATR stop loss (2.0x) is hit.
# Designed for 4h timeframe with tight entry conditions to target 75-200 total trades over 4 years.
# Works in both bull and bear markets by trading breakouts in either direction with trend filter.