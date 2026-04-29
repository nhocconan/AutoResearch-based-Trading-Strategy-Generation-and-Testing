#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d for direction and 1h for timing
# 4h Donchian(20) breakout defines the primary trend direction
# 1d EMA50 confirms higher timeframe trend alignment
# 1h RSI(14) with extreme thresholds (<20 for long, >80 for short) provides precise entry timing
# Volume confirmation (>1.5x 20-period average) filters weak breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
# Uses discrete position sizes (0.0, ±0.20) to minimize fee churn

name = "1h_Donchian20_1dEMA50_RSI_Extreme_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_20_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_20_4h)
    
    # Calculate 1d EMA50 for higher timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Warmup for 1d EMA50, 4h Donchian, 1h RSI
    
    for i in range(start_idx, n):
        # Skip if outside trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_rsi = rsi_values[i]
        curr_donchian_high = donchian_high_4h_aligned[i]
        curr_donchian_low = donchian_low_4h_aligned[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian lower OR RSI > 70 (overbought)
            if curr_close < curr_donchian_low or curr_rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian upper OR RSI < 30 (oversold)
            if curr_close > curr_donchian_high or curr_rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price above 4h Donchian upper + price above 1d EMA50 + RSI < 20 (oversold) + volume confirmation
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_1d and 
                curr_rsi < 20 and 
                vol_confirm):
                signals[i] = 0.20
                position = 1
            # Short entry: price below 4h Donchian lower + price below 1d EMA50 + RSI > 80 (overbought) + volume confirmation
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_1d and 
                  curr_rsi > 80 and 
                  vol_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals