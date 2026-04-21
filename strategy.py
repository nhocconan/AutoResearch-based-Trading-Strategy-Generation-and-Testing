#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume surge and ADX trend filter.
# Works in bull/bear: uses volatility-based pivot levels (not moving averages) and requires volume confirmation.
# Target: 20-40 trades/year by requiring strict confluence of price, volume, and trend.
# Entry: Long when price > R1 + volume surge + ADX > 20; Short when price < S1 + volume surge + ADX > 20.
# Exit: Opposite touch of S1/R1 or volume drops below average.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for pivot levels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot levels (using prior day's OHLC)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    vol_d = df_1d['volume'].values
    
    pivot_d = (high_d + low_d + close_d) / 3
    r1_d = 2 * pivot_d - low_d
    s1_d = 2 * pivot_d - high_d
    
    # Align daily data to 4h (wait for daily close)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # Calculate 14-period ADX on daily timeframe for trend strength
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # +DM and -DM
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # Skip first NaN
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation using 1d volume
    vol_ma_10_1d = pd.Series(vol_d).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_d_aligned[i]) or np.isnan(r1_d_aligned[i]) or np.isnan(s1_d_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        vol_current = align_htf_to_ltf(prices, df_1d, vol_d)[i]  # 1d volume aligned to 4h
        
        # Daily pivot levels
        r1_val = r1_d_aligned[i]
        s1_val = s1_d_aligned[i]
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        # Volume confirmation: current volume > 1.5x 10-day average
        volume_confirm = vol_current > 1.5 * vol_ma_10_1d_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R1 with volume surge in trending market
            if (trending and 
                price_close > r1_val and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume surge in trending market
            elif (trending and 
                  price_close < s1_val and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S1 OR volume drops below average
                if price_close < s1_val:
                    exit_signal = True
                elif vol_current < vol_ma_10_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above R1 OR volume drops below average
                if price_close > r1_val:
                    exit_signal = True
                elif vol_current < vol_ma_10_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSurge_ADX"
timeframe = "4h"
leverage = 1.0