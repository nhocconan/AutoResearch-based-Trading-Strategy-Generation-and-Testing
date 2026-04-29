#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x average
# Exit when price crosses the Donchian midpoint (mean of high/low over 20 periods)
# Uses 1d timeframe for structure, 1w for trend filter to reduce whipsaw in ranging markets
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian Channel (20) on 1d data
    period_donch = 20
    highest_high = pd.Series(high).rolling(window=period_donch, min_periods=period_donch).max().values
    lowest_low = pd.Series(low).rolling(window=period_donch, min_periods=period_donch).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = highest_high[i]
        curr_donchian_low = lowest_low[i]
        curr_donchian_mid = donchian_mid[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend alignment
            if curr_volume_confirm:
                # Bullish entry: price breaks above Donchian high AND above 1w EMA50
                if curr_high > curr_donchian_high and curr_close > curr_ema50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian low AND below 1w EMA50
                elif curr_low < curr_donchian_low and curr_close < curr_ema50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when price crosses Donchian midpoint
            if curr_close < curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price crosses Donchian midpoint
            if curr_close > curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals