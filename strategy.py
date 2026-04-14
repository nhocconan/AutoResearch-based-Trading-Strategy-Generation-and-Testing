#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Keltner Channel Breakout with Weekly ADX Trend Filter and Volume Spike
# Uses Keltner Channel (20, 2*ATR) for volatility-based breakout entries
# Weekly ADX (14) filters for trending markets to avoid whipsaws in ranging conditions
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly ADX data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate ADX (14) on weekly data
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # True Range
    tr1 = np.abs(high_w[1:] - low_w[1:])
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align weekly ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Load daily data for Keltner Channel calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Keltner Channel (20, 2*ATR) on daily data
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range for ATR
    tr1_d = np.abs(high_d[1:] - low_d[1:])
    tr2_d = np.abs(high_d[1:] - close_d[:-1])
    tr3_d = np.abs(low_d[1:] - close_d[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr_20_d = pd.Series(tr_d).ewm(span=20, adjust=False).mean().values
    
    # Keltner Channel
    ema_20_d = pd.Series(close_d).ewm(span=20, adjust=False).mean().values
    upper_kc = ema_20_d + 2 * atr_20_d
    lower_kc = ema_20_d - 2 * atr_20_d
    
    # Align daily Keltner Channel to 12h timeframe
    upper_kc_aligned = align_htf_to_ltf(prices, df_daily, upper_kc)
    lower_kc_aligned = align_htf_to_ltf(prices, df_daily, lower_kc)
    ema_20_d_aligned = align_htf_to_ltf(prices, df_daily, ema_20_d)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for weekly ADX and daily KC calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_kc_aligned[i]) or 
            np.isnan(lower_kc_aligned[i]) or np.isnan(ema_20_d_aligned[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper KC with volume filter and trending market
            if price > upper_kc_aligned[i] and vol > 1.5 * avg_vol[i] and trending:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower KC with volume filter and trending market
            elif price < lower_kc_aligned[i] and vol > 1.5 * avg_vol[i] and trending:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below middle line (EMA) or opposite KC
            if price < ema_20_d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above middle line (EMA) or opposite KC
            if price > ema_20_d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Keltner_Breakout_WeeklyADX_Volume"
timeframe = "12h"
leverage = 1.0