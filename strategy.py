#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume spike
# Donchian channels capture momentum breakouts; filtering by 1w EMA trend ensures
# alignment with higher timeframe direction. Volume confirmation reduces false breaks.
# Designed for low frequency (~10-25 trades/year) to minimize fee drag and improve generalization.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = np.full(len(df_1d), np.nan)
    donchian_low = np.full(len(df_1d), np.nan)
    
    for i in range(20, len(df_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 1d timeframe (wait for 1d close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA trend filter (34-period)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 1.5 x 30-day average (approx)
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (20 bars), 1w EMA (34), volume MA (30)
    start_idx = max(20, 34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1w EMA
        bullish_trend = price > ema_34_1w_aligned[i]
        bearish_trend = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and bullish trend
            if price > donchian_high_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below Donchian low with volume and bearish trend
            elif price < donchian_low_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low (mean reversion) or trend turns bearish
            if price <= donchian_low_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high (mean reversion) or trend turns bullish
            if price >= donchian_high_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0