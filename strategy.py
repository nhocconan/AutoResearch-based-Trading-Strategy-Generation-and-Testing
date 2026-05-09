#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1-day EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; values < -80 = oversold, > -20 = overbought.
# EMA50 on daily chart confirms trend direction; volume > 1.5x average confirms momentum.
# Designed for low trade frequency (20-50/year) to minimize fee drag in bear markets.
# Works in bull markets via oversold bounces and in bear markets via overbought rejections.
name = "6h_WilliamsR_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50)  # Need 14 for Williams %R and 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        ema_1d = ema_50_1d_aligned[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.5x average
            if wr < -80 and close[i] > ema_1d and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.5x average
            elif wr > -20 and close[i] < ema_1d and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -20 (overbought) OR trend reverses (price < 1d EMA50)
            if wr > -20 or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -80 (oversold) OR trend reverses (price > 1d EMA50)
            if wr < -80 or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals