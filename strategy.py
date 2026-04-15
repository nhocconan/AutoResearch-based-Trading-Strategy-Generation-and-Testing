#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation.
# Uses weekly EMA34 for trend direction (bullish when price > weekly EMA34, bearish when price < weekly EMA34).
# 6h Williams %R(14) for overbought/oversold signals: long when %R crosses above -80 from below, short when crosses below -20 from above.
# Volume confirmation: current 6h volume > 2.0x 20-period 6h volume SMA to ensure participation.
# Designed for low trade frequency (12-30/year) to minimize fee drag in ranging and trending markets.
# Works in bull markets by buying oversold dips in uptrend, and in bear markets by selling overbought rallies in downtrend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(34) for long-term trend
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if required data is NaN
        if np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 6h Williams %R Calculation ===
        # Need 14-period lookback including current bar
        if i < 13:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        
        if highest_high == lowest_low:
            williams_r = -50.0  # avoid division by zero
        else:
            williams_r = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100.0
        
        # Previous Williams %R for crossover detection
        if i < 14:
            signals[i] = 0.0
            continue
            
        prev_highest_high = np.max(high[i-14:i])
        prev_lowest_low = np.min(low[i-14:i])
        
        if prev_highest_high == prev_lowest_low:
            prev_williams_r = -50.0
        else:
            prev_williams_r = ((prev_highest_high - close[i-1]) / (prev_highest_high - prev_lowest_low)) * -100.0
        
        # Volume filter: current 6h volume > 2.0x 20-period 6h volume SMA
        if i < 20:
            vol_sma_20 = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_sma_20 = np.mean(volume[i-19:i+1])
        vol_confirm = volume[i] > (vol_sma_20 * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Weekly bullish trend: price above weekly EMA34
        # 2. Williams %R crosses above -80 from below (oversold bounce)
        # 3. Volume confirmation
        if (close[i] > ema_34_1w_aligned[i] and
            prev_williams_r < -80 and williams_r >= -80 and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Weekly bearish trend: price below weekly EMA34
        # 2. Williams %R crosses below -20 from above (overbought rejection)
        # 3. Volume confirmation
        elif (close[i] < ema_34_1w_aligned[i] and
              prev_williams_r > -20 and williams_r <= -20 and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_MeanReversion_1wEMA34_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0