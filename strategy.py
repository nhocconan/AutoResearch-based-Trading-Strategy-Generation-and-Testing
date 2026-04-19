#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with daily volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 AND volume > 1.2x daily average volume AND ADX(14) > 25 (trending market)
# Short when price breaks below S1 AND volume > 1.2x daily average volume AND ADX(14) > 25
# Exit when price returns to the previous day's pivot point
# Uses Camarilla for intraday support/resistance, volume for confirmation, ADX to avoid ranging markets.
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
    
    # Get 1d data for Camarilla pivot, volume average, and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # Using yesterday's OHLC to calculate today's levels (no look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Calculate 14-period ADX for trend strength
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 1-day average volume for confirmation (20-period MA)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_val > 25
        
        if position == 0:
            # Long entry: break above R1 + volume spike + trending market
            if price > r1_val and vol > 1.2 * vol_ma and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + trending market
            elif price < s1_val and vol > 1.2 * vol_ma and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point (mean reversion within trend)
            # Calculate current day's pivot from previous day's data
            pivot_val = (df_1d['high'].shift(1).iloc[i//2] + df_1d['low'].shift(1).iloc[i//2] + df_1d['close'].shift(1).iloc[i//2]) / 3 if i//2 < len(df_1d) else np.nan
            if not np.isnan(pivot_val) and price <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point
            pivot_val = (df_1d['high'].shift(1).iloc[i//2] + df_1d['low'].shift(1).iloc[i//2] + df_1d['close'].shift(1).iloc[i//2]) / 3 if i//2 < len(df_1d) else np.nan
            if not np.isnan(pivot_val) and price >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals