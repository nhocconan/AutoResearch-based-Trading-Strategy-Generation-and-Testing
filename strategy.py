#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Supertrend for trend direction and 1w RSI for mean reversion
# - Uses 1d HTF for Supertrend (ATR=10, mult=3.0): price above/below determines trend
# - Uses 1w HTF for RSI(14): extreme readings (>70 or <30) signal mean reversion opportunities
# - In bullish trend (price > Supertrend): look for long entries when weekly RSI < 30 (oversold)
# - In bearish trend (price < Supertrend): look for short entries when weekly RSI > 70 (overbought)
# - Volume confirmation: current 6h volume > 1.5x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_1w_supertrend_rsi_v2"
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
    tr1 = pd.Series(high_1d).rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0], raw=False)
    tr2 = abs(pd.Series(high_1d).diff())
    tr3 = abs(pd.Series(low_1d).diff())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1/10, adjust=False, min_periods=10).mean()
    
    # Basic Upper and Lower Bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr.values)
    lower_band = hl2 - (3.0 * atr.values)
    
    # Supertrend calculation
    supertrend = np.zeros(len(close_1d))
    direction = np.ones(len(close_1d))  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            if direction[i] == -1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Calculate 1w RSI(14)
    delta = pd.Series(close_1w).diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi.values)
    
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
        
        # Trend direction: 1 = uptrend, -1 = downtrend
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