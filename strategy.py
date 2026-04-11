#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with volume confirmation and 1d trend filter
# Long when price breaks above upper BB(20,2) + BB width < 20th percentile (squeeze) + volume > 1.5x avg + 1d trend up
# Short when price breaks below lower BB(20,2) + BB width < 20th percentile + volume > 1.5x avg + 1d trend down
# Exit when price returns to BB middle or squeeze ends
# Designed for 20-40 trades/year on 4h timeframe with low turnover and high edge during breakouts from low volatility

name = "4h_1d_bb_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_middle = sma_20
    bb_width = bb_upper - bb_lower
    
    # Calculate 50-period percentile rank of BB width for squeeze detection (20th percentile threshold)
    bb_width_series = pd.Series(bb_width)
    bb_width_rank = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    squeeze_condition = bb_width_rank < 0.2  # BB width in bottom 20% = squeeze
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(squeeze_condition[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: BB breakout during squeeze
        long_entry = (close[i] > bb_upper[i-1]) and squeeze_condition[i-1] and volume_filter and is_uptrend
        short_entry = (close[i] < bb_lower[i-1]) and squeeze_condition[i-1] and volume_filter and is_downtrend
        
        # Exit conditions: return to middle or squeeze ends or trend change
        long_exit = (close[i] < bb_middle[i]) or (not squeeze_condition[i]) or (not is_uptrend)
        short_exit = (close[i] > bb_middle[i]) or (not squeeze_condition[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals