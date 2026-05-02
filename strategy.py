#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR breakout + 1w/1d regime filter + volume confirmation
# Uses ATR(14) breakout from 6h Donchian(20) channels for entry
# 1w ADX > 25 and 1d ADX > 20 confirms strong trending regime (avoids chop)
# 1d EMA50 filter ensures alignment with daily trend to avoid counter-trend trades
# Volume > 1.5x 20-period EMA confirms institutional participation
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Works in bull markets (trending up + volume) and bear markets (trending down + volume)
# Uses discrete position sizing (0.25) to balance return potential with drawdown control

name = "6h_ATRBreakout_1w1dADX_1dEMA50_Trend_Volume"
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
    
    # 1w data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 1d data for trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w ADX(14) - trend strength filter
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        if len(high) >= period:
            atr[period-1] = np.mean(tr[:period])
            plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
            minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
            
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Calculate DI+ and DI-
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # Calculate ADX (smoothed DX)
        adx = np.zeros_like(high)
        if len(high) >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 1d ADX(14) - regime filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian(20) channels for breakout
    def calculate_donchian_channels(high, low, period=20):
        """Calculate Donchian channels"""
        upper = np.full_like(high, np.nan, dtype=np.float64)
        lower = np.full_like(high, np.nan, dtype=np.float64)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        
        return upper, lower
    
    upper_channel, lower_channel = calculate_donchian_channels(high, low, 20)
    
    # 6h ATR(14) for breakout confirmation
    def calculate_atr(high, low, close, period=14):
        """Calculate ATR (Average True Range)"""
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        if len(high) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_6h = calculate_atr(high, low, close, 14)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1w ADX > 25 AND 1d ADX > 20 (strong trending market)
        strong_trend_regime = (adx_1w_aligned[i] > 25) and (adx_1d_aligned[i] > 20)
        
        # Trend bias from 1d EMA50
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions with ATR confirmation
        bullish_breakout = (close[i] > upper_channel[i]) and (close[i] - upper_channel[i] > 0.5 * atr_6h[i])
        bearish_breakout = (close[i] < lower_channel[i]) and (lower_channel[i] - close[i] > 0.5 * atr_6h[i])
        
        if position == 0:  # Flat - look for new entries
            if strong_trend_regime and bullish_bias and bullish_breakout and volume_confirmation[i]:
                # Long: Strong trend regime, daily trend up, bullish breakout with ATR confirmation, volume
                signals[i] = 0.25
                position = 1
            elif strong_trend_regime and bearish_bias and bearish_breakout and volume_confirmation[i]:
                # Short: Strong trend regime, daily trend down, bearish breakout with ATR confirmation, volume
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Trend regime weakens OR trend bias turns bearish OR price breaks below lower channel
            if (not strong_trend_regime) or (not bullish_bias) or (close[i] < lower_channel[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Trend regime weakens OR trend bias turns bullish OR price breaks above upper channel
            if (not strong_trend_regime) or (not bearish_bias) or (close[i] > upper_channel[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals