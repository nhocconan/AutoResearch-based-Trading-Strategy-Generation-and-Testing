#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion strategy using 4h/1d Bollinger Bands with RSI filter
# - 4h/1d Bollinger Bands (20, 2.0) identify overextended price levels
# - RSI(14) < 30 for long, > 70 for short on 1h timeframe
# - Volume confirmation: current volume > 1.5x 20-period average
# - Session filter (08-20 UTC) to avoid low-liquidity hours
# - Fixed position size 0.20 to control drawdown
# - Mean reversion works in both bull/bear markets as price tends to revert to mean
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_bb_rsi_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 4h Bollinger Bands (20, 2.0)
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_4h = sma_20_4h + 2.0 * std_20_4h
    lower_4h = sma_20_4h - 2.0 * std_20_4h
    
    # Calculate 1d Bollinger Bands (20, 2.0)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_1d = sma_20_1d + 2.0 * std_20_1d
    lower_1d = sma_20_1d - 2.0 * std_20_1d
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with 50 (neutral)
    
    # Align all HTF data to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or
            np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]) or not in_session[i] or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 1h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Fixed position size to minimize fee churn
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit when price returns to 4h or 1d SMA (mean reversion complete)
            sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_20_4h)
            sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
            if (not np.isnan(sma_20_4h_aligned[i]) and not np.isnan(sma_20_1d_aligned[i]) and
                (close[i] >= sma_20_4h_aligned[i] or close[i] >= sma_20_1d_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price returns to 4h or 1d SMA (mean reversion complete)
            sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_20_4h)
            sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
            if (not np.isnan(sma_20_4h_aligned[i]) and not np.isnan(sma_20_1d_aligned[i]) and
                (close[i] <= sma_20_4h_aligned[i] or close[i] <= sma_20_1d_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion entry with volume confirmation
            if volume_confirmed:
                # Long when price touches/below lower BB and RSI < 30 (oversold)
                if (close[i] <= lower_4h_aligned[i] or close[i] <= lower_1d_aligned[i]) and rsi[i] < 30:
                    position = 1
                    signals[i] = position_size
                # Short when price touches/above upper BB and RSI > 70 (overbought)
                elif (close[i] >= upper_4h_aligned[i] or close[i] >= upper_1d_aligned[i]) and rsi[i] > 70:
                    position = -1
                    signals[i] = -position_size
    
    return signals