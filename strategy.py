#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike (>2.0x median) and 1d ADX trend filter (>25)
# Uses 1h timeframe for precision entries, 4h for volume confirmation, 1d for trend regime
# Long when price > R3 AND 4h volume > 2.0x 20-period 4h volume median AND 1d ADX > 25
# Short when price < S3 AND 4h volume > 2.0x 20-period 4h volume median AND 1d ADX > 25
# Exit on price returning to pivot point (PP) or ATR stoploss (1.5 ATR)
# Position size 0.20 to limit fee drag. Target: 60-150 total trades over 4 years (15-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # === 4h Indicators ===
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h volume median (20-period) for spike detection
    vol_median_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    vol_median_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20_4h)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (prev_high + prev_low + prev_close) / 3.0
    # R3 = Close + (High - Low) * 1.1 / 4
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 4.0
    # S3 = Close - (High - Low) * 1.1 / 4
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d ADX for trend filter (14-period)
    # Calculate +DI, -DI, DX
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = -np.diff(low_4h, prepend=low_4h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    tr_4h = np.maximum(tr_4h1, tr_4h2)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_4h = wilder_smooth(tr_4h, 14)
    plus_di_4h = 100 * wilder_smooth(plus_dm, 14) / atr_4h
    minus_di_4h = 100 * wilder_smooth(minus_dm, 14) / atr_4h
    dx_4h = 100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h + 1e-10)
    adx_4h = wilder_smooth(dx_4h, 14)
    
    # Align 1d ADX (using 4h ADX as proxy for daily trend)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(60, 20, 14, 2)  # 4h volume median, ATR, ADX, 1d shift
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(vol_median_20_4h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume (aligned)
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        if np.isnan(vol_4h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 2.0x 20-period 4h volume median
        vol_threshold = vol_median_20_4h_aligned[i] * 2.0
        vol_confirm = vol_4h_aligned[i] > vol_threshold
        
        # Trend filter: 1d ADX > 25 (trending market)
        trend_filter = adx_aligned[i] > 25.0
        
        # Price levels
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        pp_level = pp_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price returning to PP or ATR stoploss
            if price <= pp_level or price <= entry_price - 1.5 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price returning to PP or ATR stoploss
            if price >= pp_level or price >= entry_price + 1.5 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            price = close[i]
            
            # LONG CONDITIONS
            # Price > R3 AND volume confirmation AND trending market (ADX > 25)
            if price > r3_level and vol_confirm and trend_filter:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < S3 AND volume confirmation AND trending market (ADX > 25)
            elif price < s3_level and vol_confirm and trend_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "1h_Camarilla_R3S3_4hVolMedian2.0x_1dADX25_v1"
timeframe = "1h"
leverage = 1.0