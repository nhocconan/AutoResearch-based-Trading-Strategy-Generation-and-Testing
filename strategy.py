#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based volatility breakout with 1d trend filter and volume confirmation
# Uses ATR to dynamically size breakout thresholds, adapting to market volatility.
# Combines with 1d EMA trend filter to avoid counter-trend breakouts in choppy markets.
# Volume spike confirms institutional participation. Designed for 6h timeframe to
# reduce trade frequency and avoid fee drag while capturing medium-term trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate ATR (14-period) on 6h data
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate dynamic breakout channels using ATR
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(14, n):
        upper_channel[i] = close[i-1] + 1.5 * atr[i-1]
        lower_channel[i] = close[i-1] - 1.5 * atr[i-1]
    
    # 1d EMA trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.8 x 30-period average (7.5 hours of 6h bars)
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR (14), EMA (34), volume MA (30)
    start_idx = max(14, 34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.8 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        if position == 0:
            # Long: break above upper channel with volume and bullish trend
            if price > upper_channel[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below lower channel with volume and bearish trend
            elif price < lower_channel[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint or trend turns bearish
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if price <= midpoint or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midpoint or trend turns bullish
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if price >= midpoint or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ATR_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0