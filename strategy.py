#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining Bollinger Band breakout with 1w ADX regime filter
# In trending markets (ADX > 25): trade breakouts of Bollinger Bands (20, 2) in direction of trend
# In ranging markets (ADX < 20): mean revert at Bollinger Band extremes
# Bollinger Bands provide dynamic support/resistance that adapts to volatility
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: trend following breakouts in strong trends, mean reversion at BB extremes in ranging markets

name = "12h_1w_bb_breakout_adx_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14) for regime detection
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
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Bollinger Band parameters
    bb_period = 20
    bb_std = 2.0
    
    for i in range(bb_period, n):
        # Skip if ADX data is invalid
        if np.isnan(adx_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Bollinger Bands for current 12h bar
        close_slice = close[:i+1]
        if len(close_slice) < bb_period:
            signals[i] = 0.0
            continue
            
        sma = np.mean(close_slice[-bb_period:])
        std = np.std(close_slice[-bb_period:])
        upper_band = sma + (bb_std * std)
        lower_band = sma - (bb_std * std)
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price closes below middle band (SMA)
                if close[i] <= sma:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price returns from lower band
                if close[i] >= sma:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price closes above middle band (SMA)
                if close[i] >= sma:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price returns from upper band
                if close[i] <= sma:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Trade Bollinger Band breakouts in trending market
                if close[i] > upper_band:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lower_band:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at Bollinger Band extremes in ranging market
                if close[i] < lower_band:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > upper_band:
                    position = -1
                    signals[i] = -0.25
    
    return signals