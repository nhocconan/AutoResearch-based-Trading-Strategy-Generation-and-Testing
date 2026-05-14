#!/usr/bin/env python3
"""
6h_RSI_Stochastic_BullBear_Power_With_Volume
Hypothesis: Combines Elder Ray's Bull/Bear Power with RSI and Stochastic to identify
strong momentum in the direction of the 1d trend, filtered by volume spikes.
Works in bull markets by capturing strong uptrends and in bear markets by catching
bearish rallies or shorting weak rallies. Uses 6h for entry timing and 1d for trend.
Target: 20-40 trades/year per symbol.
"""

name = "6h_RSI_Stochastic_BullBear_Power_With_Volume"
timeframe = "6h"
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
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Stochastic(14,3,3)
    lowest_low = low_s.rolling(window=14, min_periods=14).min()
    highest_high = high_s.rolling(window=14, min_periods=14).max()
    k_percent = 100 * (close_s - lowest_low) / (highest_high - lowest_low)
    k_percent = k_percent.replace([np.inf, -np.inf], np.nan).fillna(50)
    d_percent = k_percent.rolling(window=3, min_periods=3).mean()
    stoch_k = k_percent.values
    stoch_d = d_percent.values
    
    # Bull Power and Bear Power (Elder Ray) using EMA13
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema13[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.8
        
        # Conditions for long entry
        rsi_overbought = rsi[i] > 60
        stoch_bullish_cross = stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]
        bullish_momentum = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        
        # Conditions for short entry
        rsi_oversold = rsi[i] < 40
        stoch_bearish_cross = stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]
        bearish_momentum = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # Enter long: bullish momentum + 1d uptrend + volume
            if (rsi_overbought and stoch_bullish_cross and bullish_momentum and
                trend_1d_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish momentum + 1d downtrend + volume
            elif (rsi_oversold and stoch_bearish_cross and bearish_momentum and
                  trend_1d_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when momentum fades or trend changes
            if (bull_power[i] <= 0 or rsi[i] < 50 or trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when momentum fades or trend changes
            if (bear_power[i] >= 0 or rsi[i] > 50 or trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals