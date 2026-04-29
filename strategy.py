#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume spike
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume spike
# Exit when price crosses 1d EMA10 (trailing stop proxy)
# Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Works in bull/bear markets by following 1w trend while capturing 1d momentum breakouts

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (completed 1w bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) using 1d data
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(10) for exit signal
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 10)  # warmup for EMA50, Donchian, and EMA10
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema50 = ema50_aligned[i]
        curr_ema10 = ema_10[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: break above Donchian high AND above 1w EMA50 (bullish regime)
                if curr_high > curr_donchian_high and curr_close > curr_ema50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: break below Donchian low AND below 1w EMA50 (bearish regime)
                elif curr_low < curr_donchian_low and curr_close < curr_ema50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when price crosses below EMA10
            if curr_close < curr_ema10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price crosses above EMA10
            if curr_close > curr_ema10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals