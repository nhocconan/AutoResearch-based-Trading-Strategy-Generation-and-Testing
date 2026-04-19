#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d EMA200 for long-term trend and 12h ADX for trend strength.
# Enters only when price breaks above/below 12h Donchian(10) channels with volume confirmation.
# Uses 1d/1w volatility regime filter to adapt to market conditions.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull/bear by following higher timeframe trend and avoiding choppy markets.
name = "12h_1d_EMA200_ADX14_Donchian10_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA200 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 1w data for volatility regime (ATR ratio)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1_w = np.maximum(high_1w - low_1w, 
                       np.absolute(high_1w - np.roll(close_1w, 1)),
                       np.absolute(low_1w - np.roll(close_1w, 1)))
    tr1_w[0] = high_1w[0] - low_1w[0]
    atr_10_1w = pd.Series(tr1_w).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30_1w = pd.Series(tr1_w).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ratio_1w = atr_10_1w / (atr_30_1w + 1e-10)
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    
    # Get 12h data for ADX(14) and Donchian(10)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX calculation
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(low_12h)
    plus_dm[1:] = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm[1:] = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.absolute(high_12h - np.roll(close_12h, 1)),
                        np.absolute(low_12h - np.roll(close_12h, 1)))
    tr_12h[0] = high_12h[0] - low_12h[0]
    
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_12h + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_12h + 1e-10)
    dx_14 = 100 * np.absolute(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_12h, adx_14)
    
    # Donchian channels: 10-period high/low
    high_10_12h = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    low_10_12h = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    high_10_12h_aligned = align_htf_to_ltf(prices, df_12h, high_10_12h)
    low_10_12h_aligned = align_htf_to_ltf(prices, df_12h, low_10_12h)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(high_10_12h_aligned[i]) or np.isnan(low_10_12h_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr_ratio_1w_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: avoid extremely high volatility ( ATR ratio > 1.5 )
        vol_regime_filter = atr_ratio_1w_aligned[i] <= 1.5
        
        if position == 0:
            # Long: price above 1d EMA200 AND ADX > 20 (trending) AND breaks 12h Donchian high with volume
            if (close[i] > ema_200_1d_aligned[i] and 
                adx_14_aligned[i] > 20 and 
                close[i] > high_10_12h_aligned[i] and 
                volume_filter[i] and 
                vol_regime_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA200 AND ADX > 20 (trending) AND breaks 12h Donchian low with volume
            elif (close[i] < ema_200_1d_aligned[i] and 
                  adx_14_aligned[i] > 20 and 
                  close[i] < low_10_12h_aligned[i] and 
                  volume_filter[i] and 
                  vol_regime_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA200 OR ADX < 15 (losing trend) OR breaks 12h Donchian low
            if (close[i] < ema_200_1d_aligned[i] or 
                adx_14_aligned[i] < 15 or 
                close[i] < low_10_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA200 OR ADX < 15 (losing trend) OR breaks 12h Donchian high
            if (close[i] > ema_200_1d_aligned[i] or 
                adx_14_aligned[i] < 15 or 
                close[i] > high_10_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals