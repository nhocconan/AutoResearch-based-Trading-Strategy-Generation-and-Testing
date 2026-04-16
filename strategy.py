#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels (R3/S3 fade, R4/S4 breakout)
# with volume confirmation and ADX(14) regime filter from 1d.
# Long when price breaks above R4 with volume > 1.5x average and ADX > 25 (strong trend).
# Short when price breaks below S4 with volume > 1.5x average and ADX > 25.
# Fade longs at R3 when price fails to break R4 and shows weakness (close < open).
# Fade shorts at S3 when price fails to break S4 and shows strength (close > open).
# Exit when price returns to daily pivot point (PP) or opposite Camarilla level.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for Camarilla levels and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: Camarilla Pivot Levels (based on prior day) ===
    # Calculate using prior day's high, low, close (shift by 1 to use completed day only)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Camarilla levels
    pp = (phigh + plow + pclose) / 3.0
    range_ = phigh - plow
    r3 = pp + range_ * 1.1 / 4
    s3 = pp - range_ * 1.1 / 4
    r4 = pp + range_ * 1.1 / 2
    s4 = pp - range_ * 1.1 / 2
    
    # Align daily Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === Daily Indicators: ADX (14) for trend strength filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing)
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Directional Index (DX) and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume moving average (20-period) on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to daily pivot or drops to S3
            if price <= pp_val or price <= s3_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to daily pivot or rises to R3
            if price >= pp_val or price >= r3_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: only trade when ADX > 25 (strong trend)
            trend_filter = adx_val > 25
            
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # BREAKOUT LONG: Price breaks above R4 with trend and volume confirmation
            if (price > r4_val) and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # BREAKOUT SHORT: Price breaks below S4 with trend and volume confirmation
            elif (price < s4_val) and trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
            
            # FADE LONG: Price rejects at S3 (fails to break lower) with bullish close
            elif (price >= s3_val and price <= s4_val) and trend_filter and vol_filter:
                # Look for bullish reversal: close > open and price showing strength
                if i > 0 and close[i] > prices['open'].iloc[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            
            # FADE SHORT: Price rejects at R3 (fails to break higher) with bearish close
            elif (price >= r3_val and price <= r4_val) and trend_filter and vol_filter:
                # Look for bearish reversal: close < open and price showing weakness
                if i > 0 and close[i] < prices['open'].iloc[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dCamarillaR3S3R4S4_Volume_ADXFilter_V1"
timeframe = "6h"
leverage = 1.0