#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR for volatility regime
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Get weekly data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly SMA(40) for trend filter
    if len(close_1w) >= 40:
        sma_1w = np.full_like(close_1w, np.nan)
        for i in range(40, len(close_1w)):
            sma_1w[i] = np.mean(close_1w[i-40:i])
    else:
        sma_1w = np.full_like(close_1w, np.nan)
    
    # Align all data to 4h timeframe
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    sma_1w_4h = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Volatility regime: ATR(14d) > 50th percentile of its 100-day history
    vol_regime = np.full(n, False)
    atr_lookback = 100
    
    if len(atr_1d_4h) >= atr_lookback + 50:
        for i in range(atr_lookback + 50, len(atr_1d_4h)):
            atr_slice = atr_1d_4h[i-atr_lookback:i]
            valid_atr = atr_slice[~np.isnan(atr_slice)]
            if len(valid_atr) > 0:
                percentile_50 = np.percentile(valid_atr, 50)
                vol_regime[i] = atr_1d_4h[i] > percentile_50
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    vol_confirm = volume > 1.5 * vol_ma
    
    # Trend filter: price > weekly SMA(40) for long, price < weekly SMA(40) for short
    trend_filter_long = close > sma_1w_4h
    trend_filter_short = close < sma_1w_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_lookback + 50, vol_period, 40) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_4h[i]) or np.isnan(sma_1w_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: high volatility regime + volume confirmation + bullish trend
            if vol_regime[i] and vol_confirm[i] and trend_filter_long[i]:
                signals[i] = 0.25
                position = 1
            # Short: high volatility regime + volume confirmation + bearish trend
            elif vol_regime[i] and vol_confirm[i] and trend_filter_short[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility drops OR trend turns bearish
            if not vol_regime[i] or not trend_filter_long[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility drops OR trend turns bullish
            if not vol_regime[i] or not trend_filter_short[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Volatility_Regime_Trend_Follow"
timeframe = "4h"
leverage = 1.0