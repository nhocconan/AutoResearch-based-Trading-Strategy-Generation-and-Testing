#!/usr/bin/env python3
"""
Hypothesis: 1-hour RSI(2) mean reversion with 4-hour trend filter and daily volume spike.
Long when RSI(2) < 10 and price > 4h EMA200 (uptrend filter) with daily volume > 1.5x 20-day average.
Short when RSI(2) > 90 and price < 4h EMA200 (downtrend filter) with daily volume > 1.5x 20-day average.
Exit when RSI(2) crosses 50 (mean reversion complete) or opposite RSI extreme reached.
Uses extreme short-term RSI for mean reversion in trending markets, with higher timeframe trend filter
to avoid counter-trend trades and volume spike to confirm institutional participation.
Designed for low trade frequency by requiring three simultaneous conditions.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(2) - very short term for mean reversion signals
    def rsi(close_prices, period):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    
    rsi2 = rsi(close, 2)
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # 200-period EMA on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Load daily data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-day average volume on daily timeframe
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(rsi2[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(vol_ma_20d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily volume spike: current daily volume > 1.5x 20-day average
        # Need to map current 1h bar to its corresponding daily volume
        # Find which day this 1h bar belongs to
        vol_spike = False
        if i >= 0:  # Always check, but we'll use the aligned daily volume
            # The aligned vol_ma_20d_aligned[i] gives us the 20-day MA up to the completed daily bar
            # For volume spike, we compare current 1h volume to the daily average
            # Approximate: if 1h volume > (daily 20-day MA / 24) * 1.5, consider it a spike
            daily_vol_average = vol_ma_20d_aligned[i]
            if not np.isnan(daily_vol_average):
                hourly_vol_average = daily_vol_average / 24.0
                vol_spike = volume[i] > 1.5 * hourly_vol_average
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) and price > 4h EMA200 (uptrend) with volume spike
            if rsi2[i] < 10 and close[i] > ema200_4h_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (overbought) and price < 4h EMA200 (downtrend) with volume spike
            elif rsi2[i] > 90 and close[i] < ema200_4h_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI(2) crosses 50 (mean reversion complete)
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI(2) >= 50
                if rsi2[i] >= 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI(2) <= 50
                if rsi2[i] <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_RSI2_MeanReversion_4hTrend_DailyVolume"
timeframe = "1h"
leverage = 1.0