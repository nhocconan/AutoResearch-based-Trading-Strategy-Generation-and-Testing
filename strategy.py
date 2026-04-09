#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1d Supertrend for trend direction and 1w RSI for mean reversion timing
# - Uses 1d HTF for Supertrend (ATR=10, mult=3.0): determines primary trend direction
# - Uses 1w HTF for RSI(14): extreme readings (>70 or <30) signal mean reversion entries
# - In bullish 1d trend (price > Supertrend): look for long entries when weekly RSI < 30 (oversold pullback)
# - In bearish 1d trend (price < Supertrend): look for short entries when weekly RSI > 70 (overbought bounce)
# - Volume confirmation: current 6h volume > 1.5x 20-period average to filter low-quality signals
# - Fixed position size 0.25 to manage drawdown through 2022-like crashes
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: trend filter prevents counter-trend trading in strong moves,
#   while RSI extremes provide mean reversion entries within the trend

name = "6h_1d_1w_supertrend_rsi_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Supertrend (ATR=10, mult=3.0)
    # True Range
    tr1 = pd.Series(high_1d).rolling(window=2).max().values - pd.Series(low_1d).rolling(window=2).min().values
    tr2 = np.abs(pd.Series(high_1d).rolling(window=2).shift(1).values - pd.Series(close_1d).rolling(window=2).shift(1).values)
    tr3 = np.abs(pd.Series(low_1d).rolling(window=2).shift(1).values - pd.Series(close_1d).rolling(window=2).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan, dtype=float)
    direction = np.full_like(close_1d, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    # Start calculation after warmup period
    for i in range(10, len(close_1d)):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(close_1d[i]):
            continue
            
        if i == 10:
            # Initialize first values
            supertrend[i] = upper_band[i]
            direction[i] = 1  # Start with uptrend assumption
        else:
            prev_supertrend = supertrend[i-1]
            prev_direction = direction[i-1]
            
            if np.isnan(prev_supertrend) or np.isnan(prev_direction):
                supertrend[i] = upper_band[i]
                direction[i] = 1
                continue
            
            # Supertrend logic
            if prev_direction == 1:  # Was in uptrend
                if close_1d[i] <= prev_supertrend:
                    # Trend change to downtrend
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    # Stay in uptrend
                    supertrend[i] = max(prev_supertrend, lower_band[i])
                    direction[i] = 1
            else:  # Was in downtrend
                if close_1d[i] >= prev_supertrend:
                    # Trend change to uptrend
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    # Stay in downtrend
                    supertrend[i] = min(prev_supertrend, upper_band[i])
                    direction[i] = -1
    
    # Calculate 1w RSI (14 periods)
    # RSI = 100 - (100 / (1 + RS)) where RS = average gain / average loss
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend direction from 1d Supertrend
        bullish_trend = direction_aligned[i] == 1
        bearish_trend = direction_aligned[i] == -1
        
        # RSI extremes: <30 = oversold, >70 = overbought
        oversold = rsi_aligned[i] < 30
        overbought = rsi_aligned[i] > 70
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend:
                # In bullish trend: exit when overbought or trend changes to bearish
                if overbought or bearish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Not in bullish trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_trend:
                # In bearish trend: exit when oversold or trend changes to bullish
                if oversold or bullish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Not in bearish trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend and RSI extremes
            if volume_confirmed:
                if bullish_trend and oversold:
                    # In bullish trend, weekly oversold: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and overbought:
                    # In bearish trend, weekly overbought: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals