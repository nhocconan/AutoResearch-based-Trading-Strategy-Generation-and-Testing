# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 4H_RSI_Divergence_Volume_Trend
# Hypothesis: Uses RSI divergence with price action combined with volume confirmation and 1d trend filter.
# Enters long on bullish RSI divergence (price makes lower low, RSI makes higher low) with volume spike in uptrend.
# Enters short on bearish RSI divergence (price makes higher high, RSI makes lower high) with volume spike in downtrend.
# Exits when RSI crosses 50 in opposite direction or trend reverses.
# Designed to work in both bull and bear markets by following 1d trend and using RSI exhaustion signals.
# Targets 20-40 trades per year on 4h timeframe with discrete position sizing (0.25) to minimize churn.

name = "4H_RSI_Divergence_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate RSI(14) on 4h chart
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # Warmup for EMA, volume MA, and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # RSI levels
        rsi_now = rsi_values[i]
        rsi_prev = rsi_values[i-1]
        
        # Price action for divergence detection
        price_now = close[i]
        price_prev = close[i-1]
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Look back 3 bars for swing points
            if i >= 3:
                price_low_3 = np.min(low[i-3:i+1])
                price_low_now = low[i]
                rsi_low_3 = np.min(rsi_values[i-3:i+1])
                rsi_low_now = rsi_values[i]
                
                bullish_div = (price_low_now < price_low_3 and 
                              rsi_low_now > rsi_low_3 and
                              rsi_now < 40)  # Only in oversold territory
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                price_high_3 = np.max(high[i-3:i+1])
                price_high_now = high[i]
                rsi_high_3 = np.max(rsi_values[i-3:i+1])
                rsi_high_now = rsi_values[i]
                
                bearish_div = (price_high_now > price_high_3 and
                              rsi_high_now < rsi_high_3 and
                              rsi_now > 60)  # Only in overbought territory
                
                # Long entry: bullish divergence in uptrend with volume spike
                if (bullish_div and 
                    price_above_ema and 
                    volume[i] > vol_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                # Short entry: bearish divergence in downtrend with volume spike
                elif (bearish_div and 
                      price_below_ema and 
                      volume[i] > vol_threshold[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: RSI crosses above 50 or trend reverses
            if (rsi_now > 50 and rsi_prev <= 50) or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses below 50 or trend reverses
            if (rsi_now < 50 and rsi_prev >= 50) or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals