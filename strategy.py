#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with weekly volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 AND weekly volume > 1.8x 4-week average AND ADX > 25 (trending market)
# Short when price breaks below S1 AND weekly volume > 1.8x 4-week average AND ADX > 25
# Exit when price crosses back through the Camarilla midpoint (close of previous day)
# Uses Camarilla for intraday pivot structure, weekly volume for conviction, ADX to avoid chop.
# Target: 15-25 trades/year per symbol.
name = "12h_Camarilla_R1S1_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume confirmation
    df_weekly = get_htf_data(prices, '1w')
    vol_ma_weekly = pd.Series(df_weekly['volume']).rolling(window=4, min_periods=4).mean().values
    vol_ma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_weekly)
    
    # Get daily data for Camarilla pivot and ADX
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (based on previous day)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    prev_close = df_daily['close'].shift(1).values
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    camarilla_mid = prev_close  # Using close as pivot point
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_daily, camarilla_mid)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1)
    
    # Calculate ADX(14) on daily timeframe for trend filter
    # True Range
    tr1 = df_daily['high'] - df_daily['low']
    tr2 = np.abs(df_daily['high'] - df_daily['close'].shift(1))
    tr3 = np.abs(df_daily['low'] - df_daily['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = df_daily['high'].diff(1).values
    down_move = df_daily['low'].diff(1).values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_weekly_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_weekly_aligned[i]
        vol = volume[i]
        mid = camarilla_mid_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        adx_val = adx_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long entry: break above R1 + volume spike + trend
            if price > r1 and vol > 1.8 * vol_ma and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + trend
            elif price < s1 and vol > 1.8 * vol_ma and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals