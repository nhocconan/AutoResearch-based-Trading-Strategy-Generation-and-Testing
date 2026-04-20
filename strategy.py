#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE for 1d indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR(14) for volatility and stop
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily 200-day SMA for trend filter
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Daily 20-day volume average for volume filter
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 1d timeframe (no shift needed since we use previous day's close)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Load weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly 50-period SMA for long-term trend
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # 1d price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after 200-day SMA warmup
        # Skip if NaN in indicators
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(sma_200_1d_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(sma_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Use previous day's values to avoid look-ahead
        atr_val = atr_14_1d_aligned[i-1] if i > 0 else atr_14_1d_aligned[i]
        sma_200_val = sma_200_1d_aligned[i-1] if i > 0 else sma_200_1d_aligned[i]
        vol_avg_val = vol_avg_20_1d_aligned[i-1] if i > 0 else vol_avg_20_1d_aligned[i]
        sma_50_1w_val = sma_50_1w_aligned[i-1] if i > 0 else sma_50_1w_aligned[i]
        price = close[i-1] if i > 0 else close[i]  # Previous day's close for signal
        vol = volume[i-1] if i > 0 else volume[i]
        
        # Trend filter: price above/below 200-day SMA
        uptrend = price > sma_200_val
        downtrend = price < sma_200_val
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = sma_50_1w_val > sma_50_1w_aligned[max(0, i-5)] if i >= 5 else True  # Rising weekly SMA
        weekly_downtrend = sma_50_1w_val < sma_50_1w_aligned[max(0, i-5)] if i >= 5 else True  # Falling weekly SMA
        
        # Volume filter: current volume above 20-day average
        vol_filter = vol > vol_avg_val
        
        # Volatility filter: only trade when volatility is elevated (breakout conditions)
        vol_regime = atr_val > np.nanpercentile(atr_14_1d_aligned[max(0, i-20):i], 70) if i >= 20 else False
        
        if position == 0:
            # Long: price breaks above 200-day SMA with volume and volatility
            if uptrend and vol_filter and vol_regime and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 200-day SMA with volume and volatility
            elif downtrend and vol_filter and vol_regime and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 200-day SMA or volatility drops
            if price <= sma_200_val or atr_val < np.nanpercentile(atr_14_1d_aligned[max(0, i-10):i], 30) if i >= 10 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 200-day SMA or volatility drops
            if price >= sma_200_val or atr_val < np.nanpercentile(atr_14_1d_aligned[max(0, i-10):i], 30) if i >= 10 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_SMA200_Breakout_VolumeVolatility_WeeklyTrend"
timeframe = "1d"
leverage = 1.0