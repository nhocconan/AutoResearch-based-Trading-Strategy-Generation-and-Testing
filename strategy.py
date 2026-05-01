#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d ADX > 25 ensures we only trade in trending markets to avoid chop whipsaws
# Volume confirmation (> 1.5x 20-period EMA) ensures institutional participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Works in bull markets via pullback longs in uptrends and bear markets via bounce shorts in downtrends
# BTC/ETH focused: avoids SOL-only bias by requiring trend alignment and volume confirmation

name = "6h_WilliamsR_1dADX25_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter (period=14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    tr_smooth = wilders_smooth(tr, period_adx)
    plus_dm_smooth = wilders_smooth(plus_dm, period_adx)
    minus_dm_smooth = wilders_smooth(minus_dm, period_adx)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smooth(dx, period_adx)
    
    # Align 1d ADX to 6h timeframe (wait for 1d bar close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R on 6h (period=14)
    def williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.values
    
    wr = williams_r(high, low, close, period=14)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (trending if ADX > 25)
        trending_market = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if trending_market:
                # Long: Williams %R oversold (< -80) with volume spike
                if wr[i] < -80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought (> -20) with volume spike
                elif wr[i] > -20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid choppy/non-trending markets
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (momentum fading) or ADX drops below 20
            if wr[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (momentum fading) or ADX drops below 20
            if wr[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals