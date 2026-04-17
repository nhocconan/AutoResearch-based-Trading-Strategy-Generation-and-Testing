#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1-week RSI filter and volume confirmation.
Trade pullbacks to the 21-period EMA in the direction of the weekly trend (RSI > 50 for long, RSI < 50 for short).
Use volume spike (>1.5x 20-day average) to confirm momentum.
Designed to work in bull markets via trend-following pullbacks and in bear via mean-reversion at EMA support/resistance.
Target: 30-100 total trades over 4 years (7-25/year).
"""
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
    
    # Get 1w data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 14-period RSI on weekly
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Get 1d data for EMA and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 21-period EMA on daily
    ema_21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 20-day average volume on daily
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to daily
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14)
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from weekly RSI
        bullish_trend = rsi_14_aligned[i] > 50
        bearish_trend = rsi_14_aligned[i] < 50
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (vol_ma_20_aligned[i] * 1.5)
        
        if position == 0:
            # Long: pullback to EMA in bullish trend with volume
            if (low[i] <= ema_21_aligned[i] <= high[i] and bullish_trend and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: pullback to EMA in bearish trend with volume
            elif (low[i] <= ema_21_aligned[i] <= high[i] and bearish_trend and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes above EMA or trend turns bearish
            if close[i] > ema_21_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes below EMA or trend turns bullish
            if close[i] < ema_21_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wRSI_EMA21_Pullback_VolumeFilter"
timeframe = "1d"
leverage = 1.0