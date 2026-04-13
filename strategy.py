# [Experiment 40787] 6h_1d_1w_Triple_Timeframe_Confluence
# Hypothesis: Uses weekly trend direction (EMA crossover), daily momentum (RSI divergence), and 6h breakout (Donchian breakout) for high-probability entries.
# Weekly EMA(50) above EMA(200) defines bullish trend, below defines bearish.
# Daily RSI(14) divergence with price (bullish: higher low in RSI with lower low in price) signals reversal in trend direction.
# 6h Donchian(20) breakout in direction of weekly trend with daily RSI confirmation.
# Works in bull markets via trend continuation and bear markets via counter-trend reversals at extremes.
# Target: 20-40 trades/year on 6h (80-160 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) and EMA(200) for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_bullish = ema_50_1w > ema_200_1w  # True for bullish trend
    
    # Get daily data for RSI divergence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Get 6h data for Donchian breakout
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Donchian(20) channels
    donch_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align all signals to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_6h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_6h, donch_low_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(weekly_bullish_aligned[i]) or \
           np.isnan(rsi_1d_aligned[i]) or \
           np.isnan(donch_high_20_aligned[i]) or \
           np.isnan(donch_low_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate daily RSI divergence (requires lookback)
        if i >= 10:  # Need at least 10 days of data for divergence
            # Get indices for daily lookback
            lookback = 10
            rsi_slice = rsi_1d_aligned[max(0, i-lookback):i+1]
            price_slice = close[max(0, i-lookback):i+1]
            
            if len(rsi_slice) >= lookback and len(price_slice) >= lookback:
                # Find local minima in price and RSI
                price_min_idx = np.argmin(price_slice)
                rsi_min_idx = np.argmin(rsi_slice)
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                bullish_div = (price_min_idx == len(price_slice)-1 and  # Recent low
                              rsi_slice[-1] > rsi_slice[0] and        # RSI higher than period start
                              price_slice[-1] < price_slice[0])       # Price lower than period start
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                price_max_idx = np.argmax(price_slice)
                rsi_max_idx = np.argmax(rsi_slice)
                bearish_div = (price_max_idx == len(price_slice)-1 and   # Recent high
                              rsi_slice[-1] < rsi_slice[0] and          # RSI lower than period start
                              price_slice[-1] > price_slice[0])         # Price higher than period start
            else:
                bullish_div = False
                bearish_div = False
        else:
            bullish_div = False
            bearish_div = False
        
        # Entry logic
        weekly_trend = weekly_bullish_aligned[i]  # True = bullish, False = bearish
        price = close[i]
        
        # Long conditions: weekly bullish AND (price breaks above Donchian high OR bullish divergence)
        if weekly_trend:
            if price > donch_high_20_aligned[i] or bullish_div:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Exit long: price breaks below Donchian low in bullish trend
            elif price < donch_low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else 0.0
        # Short conditions: weekly bearish AND (price breaks below Donchian low OR bearish divergence)
        else:
            if price < donch_low_20_aligned[i] or bearish_div:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Exit short: price breaks above Donchian high in bearish trend
            elif price > donch_high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size if position == -1 else 0.0
    
    return signals

name = "6h_1d_1w_Triple_Timeframe_Confluence"
timeframe = "6h"
leverage = 1.0