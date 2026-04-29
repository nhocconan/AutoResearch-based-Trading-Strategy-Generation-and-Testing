#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 6h Donchian upper band AND price > 12h EMA50 AND volume > 1.8x 20-period average
# Short when price breaks below 6h Donchian lower band AND price < 12h EMA50 AND volume > 1.8x 20-period average
# Exit when price crosses the opposite Donchian band (lower band for longs, upper band for shorts)
# Uses discrete position sizing (0.25) to manage drawdown and minimize fee churn
# Target: 12-25 trades/year on 6h timeframe (~50-100 total over 4 years) to stay within fee drag limits
# Works in bull markets via long breakouts with 12h uptrend
# Works in bear markets via short breakdowns with 12h downtrend
# Volume confirmation reduces false breakouts in choppy markets

name = "6h_Donchian_Breakout_12hEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=1).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_vol_spike = vol_spike[i]
        
        # Skip if Donchian levels are not available
        if np.isnan(curr_donchian_upper) or np.isnan(curr_donchian_lower):
            signals[i] = 0.0
            continue
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit when price crosses below Donchian lower band
            if curr_close < curr_donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price crosses above Donchian upper band
            if curr_close > curr_donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band AND price > 12h EMA50 AND volume spike
            if curr_close > curr_donchian_upper and curr_close > curr_ema_12h and curr_vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band AND price < 12h EMA50 AND volume spike
            elif curr_close < curr_donchian_lower and curr_close < curr_ema_12h and curr_vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals