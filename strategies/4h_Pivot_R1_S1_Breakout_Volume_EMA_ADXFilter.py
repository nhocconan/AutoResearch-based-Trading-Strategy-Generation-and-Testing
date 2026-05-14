#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Pivot Points (Camarilla style) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate pivot points from previous day
    pp = (df_1d['high'].shift(1) + df_1d['low'].shift(1) + df_1d['close'].shift(1)) / 3
    r1 = pp + (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 6
    s1 = pp - (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 6
    
    # Align pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1d EMA Trend Filter ===
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d.values)
    
    # === 4h ADX for Trend Strength Filter ===
    # Calculate ADX on 4h data
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed ATR, +DM, -DM
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / np.where(atr == 0, np.nan, atr)
    minus_di = 100 * minus_dm_smooth / np.where(atr == 0, np.nan, atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 4h timeframe (already 4h, but need to align to lower timeframe)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50  # Need EMA50 and ADX warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema50 = ema_50_aligned[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC: Close position when price returns to pivot zone ===
        if position == 1:  # Long position
            # Exit when price crosses back below R1 (failed breakout)
            if price < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above S1 (failed breakdown)
            if price > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Strong trend filter: only trade when ADX > 25 (trending market)
            if adx_val > 25:
                # LONG: Price breaks above R1 with volume confirmation and price > EMA50 (bullish bias)
                if price > r1_aligned[i] and vol_spike and price > ema50:
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below S1 with volume confirmation and price < EMA50 (bearish bias)
                elif price < s1_aligned[i] and vol_spike and price < ema50:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_EMA_ADXFilter"
timeframe = "4h"
leverage = 1.0