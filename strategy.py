#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly 52-week high breakout with volume confirmation and ADX trend filter
# Targets long-term trends in BTC/ETH/SOL by buying breakouts of annual resistance
# Works in bull markets (continuation) and bear markets (mean reversion off extreme levels)
# Weekly timeframe reduces noise, daily provides entry precision
name = "1d_Weekly52W_HighBreakout_Volume_ADXFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:  # Need ~1 year for 52-week high
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for 52-week high and ADX
    df_1w = get_htf_data(prices, '1w')
    
    # 52-week high (252 trading days ≈ 52 weeks)
    # Using weekly high over 52 periods
    weekly_high = df_1w['high'].values
    high_52w = pd.Series(weekly_high).rolling(window=52, min_periods=52).max().values
    
    # ADX(14) for trend strength
    # Calculate +DM, -DM, TR
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (equivalent to RMA)
    def rma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    plus_dm_smooth = rma(np.concatenate([[0], plus_dm]), 14)
    minus_dm_smooth = rma(np.concatenate([[0], minus_dm]), 14)
    tr_smooth = rma(np.concatenate([[0], tr]), 14)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = rma(dx, 14)
    
    # Align weekly indicators to daily
    high_52w_aligned = align_htf_to_ltf(prices, df_1w, high_52w)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 250  # Wait for 52-week high calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_52w_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_52w_val = high_52w_aligned[i]
        adx_val = adx_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above 52-week high with volume and ADX > 20 (trending market)
            if close_val > high_52w_val and vol_filter and adx_val > 20:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit: close below 50-day MA or ADX drops below 15 (trend weakening)
            ma_50 = pd.Series(close[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1]
            if close_val < ma_50 or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals