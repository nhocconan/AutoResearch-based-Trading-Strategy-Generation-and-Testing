#!/usr/bin/env python3
# 1h_rsi_sma_trend_filter_v1
# Hypothesis: Use 4h trend filter (SMA50) with 1h RSI pullback entries.
# In 4h uptrend: go long when RSI(14) pulls back to 40-45 and price > SMA20.
# In 4h downtrend: go short when RSI(14) bounces to 55-60 and price < SMA20.
# Exit when RSI reaches opposite extreme (60 for long, 40 for short) or trend flips.
# Uses 1d volume filter to avoid low-volatility chop.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_sma_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h SMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    sma50_4h = pd.Series(df_4h['close']).rolling(window=50, min_periods=50).mean().values
    sma50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma50_4h)
    
    # 1d volume filter: volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h SMA20 for dynamic support/resistance
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(sma20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: avoid low-volatility chop
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i] if vol_ma_20_1d_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: RSI >= 60 (overbought) or trend flips (price < 4h SMA50)
            if rsi[i] >= 60 or close[i] < sma50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI <= 40 (oversold) or trend flips (price > 4h SMA50)
            if rsi[i] <= 40 or close[i] > sma50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: 4h uptrend + RSI pullback to 40-45 + price > SMA20 + volume filter
            if (close[i] > sma50_4h_aligned[i] and 
                40 <= rsi[i] <= 45 and 
                close[i] > sma20[i] and 
                vol_filter):
                position = 1
                signals[i] = 0.20
            # Short entry: 4h downtrend + RSI bounce to 55-60 + price < SMA20 + volume filter
            elif (close[i] < sma50_4h_aligned[i] and 
                  55 <= rsi[i] <= 60 and 
                  close[i] < sma20[i] and 
                  vol_filter):
                position = -1
                signals[i] = -0.20
    
    return signals