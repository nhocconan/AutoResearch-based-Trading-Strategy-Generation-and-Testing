#!/usr/bin/env python3
name = "4h_Triple_SMA_Cross_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily trend filter: EMA(34) on daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Daily EMA(34) - trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily EMA(89) - stronger trend filter
    ema_89_1d = pd.Series(df_1d['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # 4h SMA(20) - short-term trend
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # 4h SMA(50) - medium-term trend
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # 4h SMA(200) - long-term trend
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 200)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_89_1d_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: daily EMA(34) > EMA(89) for uptrend, < for downtrend
        daily_uptrend = ema_34_1d_aligned[i] > ema_89_1d_aligned[i]
        daily_downtrend = ema_34_1d_aligned[i] < ema_89_1d_aligned[i]
        
        # Volume condition: above average
        vol_condition = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: SMA(20) > SMA(50) > SMA(200) with daily uptrend and volume
            if (sma_20[i] > sma_50[i] > sma_200[i] and daily_uptrend and vol_condition):
                signals[i] = 0.25
                position = 1
            # Short: SMA(20) < SMA(50) < SMA(200) with daily downtrend and volume
            elif (sma_20[i] < sma_50[i] < sma_200[i] and daily_downtrend and vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: SMA(20) < SMA(50) or daily trend reverses
            if sma_20[i] < sma_50[i] or not daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: SMA(20) > SMA(50) or daily trend reverses
            if sma_20[i] > sma_50[i] or not daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Triple SMA Cross with Daily Trend Filter
# - Uses SMA(20/50/200) on 4h for trend identification and entry signals
# - Daily EMA(34/89) as higher timeframe trend filter to avoid counter-trend trades
# - Volume confirmation to ensure institutional participation
# - Long when SMA(20) > SMA(50) > SMA(200) with daily uptrend and volume
# - Short when SMA(20) < SMA(50) < SMA(200) with daily downtrend and volume
# - Exit when short-term SMA crosses below/above medium-term SMA or daily trend reverses
# - Position size 0.25 limits risk while capturing trends
# - Designed to work in both bull (catch uptrends) and bear (catch downtrends) markets
# - Target: 20-40 trades/year to minimize fee drag while maintaining edge