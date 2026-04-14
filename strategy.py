#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour volume-weighted average price (VWAP) deviation with 12-hour trend filter and volume confirmation
# Long when price is significantly below VWAP (mean reversion) with 12-hour bullish trend and volume spike
# Short when price is significantly above VWAP (mean reversion) with 12-hour bearish trend and volume spike
# Exit when price returns to VWAP or trend changes
# Uses 12-hour EMA trend filter to align with higher timeframe momentum
# Target: 15-30 trades per year per symbol to minimize fee drag while capturing mean reversion edges
# VWAP mean reversion works in both bull and bear markets as price tends to revert to average

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    # Calculate 12h EMA for trend filter (21-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    ema_12h = pd.Series(typical_price_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 4h volume average (20-period)
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_4h, vwap)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        vwap_val = vwap_aligned[i]
        
        # Calculate deviation from VWAP as percentage
        if vwap_val != 0:
            deviation = (price - vwap_val) / vwap_val
        else:
            deviation = 0
        
        if position == 0:
            # Long setup: price below VWAP (mean reversion long) with 12h bullish trend and volume spike
            if (deviation < -0.015 and  # Price more than 1.5% below VWAP
                price > ema_12h_aligned[i] and  # Price above 12h EMA for bullish trend
                vol_4h_current > 2.0 * vol_ma_4h_aligned[i]):  # Volume spike (2x average)
                position = 1
                signals[i] = position_size
            # Short setup: price above VWAP (mean reversion short) with 12h bearish trend and volume spike
            elif (deviation > 0.015 and  # Price more than 1.5% above VWAP
                  price < ema_12h_aligned[i] and  # Price below 12h EMA for bearish trend
                  vol_4h_current > 2.0 * vol_ma_4h_aligned[i]):  # Volume spike (2x average)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or trend turns bearish
            if (deviation >= -0.005 or  # Price back near VWAP (within 0.5%)
                price < ema_12h_aligned[i]):  # Trend turned bearish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP or trend turns bullish
            if (deviation <= 0.005 or  # Price back near VWAP (within 0.5%)
                price > ema_12h_aligned[i]):  # Trend turned bullish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VWAP_MeanReversion_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0