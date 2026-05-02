#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend + volume confirmation
# Uses Williams Alligator (JAW=TEETH=LIPS SMMA) from 6h to identify trending vs ranging markets
# 1d EMA50 ensures alignment with long-term trend to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets (trend following with Alligator) and bear markets (mean reversion at extremes)

name = "6h_Williams_Alligator_1dEMA50_Volume"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Trending market: Alligator lines are separated and ordered
            # Bullish: Lips > Teeth > Jaw (green > red > blue)
            # Bearish: Lips < Teeth < Jaw (green < red < blue)
            bullish_trend = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_trend = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long entry: Bullish trend AND price > 1d EMA50 AND volume spike
            if (bullish_trend and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish trend AND price < 1d EMA50 AND volume spike
            elif (bearish_trend and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Market becomes ranging (Alligator lines intertwine) OR price below 1d EMA50
            # Alligator is sleeping when lines are close together (market ranging)
            jaw_to_lips = abs(jaw[i] - lips[i])
            teeth_to_lips = abs(teeth[i] - lips[i])
            avg_price = (high[i] + low[i]) / 2
            ranging_market = (jaw_to_lips < avg_price * 0.01) and (teeth_to_lips < avg_price * 0.01)
            
            if ranging_market or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Market becomes ranging (Alligator lines intertwine) OR price above 1d EMA50
            jaw_to_lips = abs(jaw[i] - lips[i])
            teeth_to_lips = abs(teeth[i] - lips[i])
            avg_price = (high[i] + low[i]) / 2
            ranging_market = (jaw_to_lips < avg_price * 0.01) and (teeth_to_lips < avg_price * 0.01)
            
            if ranging_market or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals