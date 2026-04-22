#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Squeeze + 1d ADX Trend Filter + Volume Confirmation
# Bollinger Squeeze identifies low volatility periods (BB width < 20th percentile) that precede breakouts.
# 1d ADX > 25 confirms strong trend direction to avoid false breakouts in ranging markets.
# Volume > 1.5x 20-period average confirms breakout strength.
# Works in both bull and bear markets by only taking breakouts in the direction of the 1d trend.
# Uses discrete position sizing (0.25) to minimize fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        dx[:] = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Bollinger Bands (20, 2) on 12h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = (upper - lower) / sma  # Normalized width
    
    # Bollinger Squeeze: BB width < 20th percentile of last 50 periods
    def calculate_percentile(arr, percentile):
        sorted_arr = np.sort(arr[~np.isnan(arr)])
        if len(sorted_arr) == 0:
            return 0
        idx = int(len(sorted_arr) * percentile / 100)
        return sorted_arr[min(idx, len(sorted_arr)-1)]
    
    bb_width_percentile = np.full_like(bb_width, np.nan)
    lookback = 50
    for i in range(lookback, len(bb_width)):
        bb_width_percentile[i] = calculate_percentile(bb_width[i-lookback:i], 20)
    
    squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    min_idx = max(bb_period, lookback)
    for i in range(min_idx, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout from squeeze in direction of 1d trend
            bullish_trend = adx_14_1d_aligned[i] > 25
            bearish_trend = adx_14_1d_aligned[i] > 25  # ADX > 25 indicates strong trend (either direction)
            
            # Long breakout: squeeze + price above upper BB + volume spike + bullish trend
            if squeeze[i] and close[i] > upper[i] and volume[i] > 1.5 * vol_avg_20[i] and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short breakout: squeeze + price below lower BB + volume spike + bearish trend
            elif squeeze[i] and close[i] < lower[i] and volume[i] > 1.5 * vol_avg_20[i] and bearish_trend:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of Bollinger Bands or volatility expands (squeeze ends)
            if position == 1:
                # Exit long: price back below middle BB or squeeze ends
                if close[i] < sma[i] or not squeeze[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price back above middle BB or squeeze ends
                if close[i] > sma[i] or not squeeze[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_BollingerSqueeze_1dADX_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0