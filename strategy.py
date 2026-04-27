# 6h Institutional Flow Detector - Detects institutional accumulation/distribution
# Uses volume-weighted price action relative to prior day VWAP to identify smart money flows
# Combines with 1d trend filter and volume confirmation for high-probability entries
# Designed for 6h timeframe to avoid overtrading while capturing multi-day moves
# Works in bull/bear by following institutional flow direction aligned with higher timeframe trend

#!/usr/bin/env python3
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
    
    # Get 1d data for VWAP calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(vwap_numerator, np.nan), 
                        where=vwap_denominator!=0)
    
    # Calculate institutional flow strength: (close - VWAP) / ATR
    # Normalizes price deviation from VWAP by volatility
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(atr_period-1, len(df_1d)):
        atr_1d[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Institutional flow: positive = buying pressure, negative = selling pressure
    flow_strength = np.divide((close_1d - vwap_1d), atr_1d,
                              out=np.full_like(close_1d, np.nan),
                              where=atr_1d!=0)
    
    # Smooth flow signal to reduce noise
    flow_smooth = np.full(len(df_1d), np.nan)
    for i in range(4, len(df_1d)):  # 5-period smoothing
        flow_smooth[i] = np.nanmean(flow_strength[i-4:i+1])
    
    # Align flow to 6h timeframe
    flow_aligned = align_htf_to_ltf(prices, df_1d, flow_smooth)
    
    # 1d EMA trend filter (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.8 x 24-period average (3 days of 6h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d VWAP (1 bar), EMA (50), volume MA (24), flow smoothing (4)
    start_idx = max(1, 50, 24, 4)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(flow_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume surge
        vol_filter = vol_now > 1.8 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        # Institutional flow signals
        strong_buying = flow_aligned[i] > 0.8   # Strong buying pressure
        strong_selling = flow_aligned[i] < -0.8 # Strong selling pressure
        
        if position == 0:
            # Long: institutional buying + volume + bullish trend
            if strong_buying and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: institutional selling + volume + bearish trend
            elif strong_selling and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: flow turns negative or trend breaks
            if flow_aligned[i] < -0.2 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: flow turns positive or trend breaks
            if flow_aligned[i] > 0.2 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Institutional_Flow_Detector_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0