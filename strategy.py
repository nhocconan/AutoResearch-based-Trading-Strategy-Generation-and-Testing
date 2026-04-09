#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining Donchian(20) breakout with 1w ADX regime filter and volume confirmation
# Donchian breakout captures momentum in trending markets
# 1w ADX > 25 filters for strong trending regime (avoids whipsaws in ranging markets)
# Volume confirmation requires current volume > 1.5 * 20-period average to avoid false breakouts
# In trending regime (ADX > 25): long when price breaks above Donchian(20) upper band, short when breaks below lower band
# Exit when price returns to Donchian(20) middle band or ADX drops below 20 (regime change)
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: trend following in strong trends, avoids trading in weak/choppy markets

name = "12h_1w_donchian_breakout_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = wilders_smoothing(tr, period)
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Middle band
    middle_band = (highest_high + lowest_low) / 2
    
    # Calculate volume average (20-period)
    volume_avg = np.full(n, np.nan)
    for i in range(lookback-1, n):
        volume_avg[i] = np.mean(volume[i-lookback+1:i+1])
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_confirmed = volume > (1.5 * volume_avg)
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_avg[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit long if price returns to middle band or regime changes to ranging
            if close[i] <= middle_band[i] or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if price returns to middle band or regime changes to ranging
            if close[i] >= middle_band[i] or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only enter in trending regime with volume confirmation
            if trending_regime and volume_confirmed[i]:
                # Long when price breaks above upper band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short when price breaks below lower band
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals