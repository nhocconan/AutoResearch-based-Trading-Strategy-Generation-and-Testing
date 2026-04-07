#!/usr/bin/env python3
"""
6h_rsi_divergence_1w_volume_v1
Hypothesis: RSI divergence on 6h combined with weekly trend filter and volume confirmation works in both bull and bear markets.
- Bullish divergence: price makes lower low, RSI makes higher low → long when price breaks above recent high with volume
- Bearish divergence: price makes higher high, RSI makes lower high → short when price breaks below recent low with volume
- Weekly trend filter: only take longs when price above weekly EMA50, shorts when below
- Volume confirmation: current volume above 20-period average
Designed for 15-30 trades/year on 6h timeframe with high-probability reversal setups.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_divergence_1w_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Higher timeframe trend filter
        above_weekly_ema50 = close[i] > ema50_1w_aligned[i]
        below_weekly_ema50 = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish divergence or price breaks below recent low
            lookback = min(10, i)
            recent_low = np.min(low[i-lookback:i+1])
            recent_rsi_low = np.min(rsi[i-lookback:i+1])
            if i >= lookback + 1:
                prev_low = np.min(low[i-lookback-1:i])
                prev_rsi_low = np.min(rsi[i-lookback-1:i])
                bearish_divergence = (low[i] > prev_low) and (rsi[i] < prev_rsi_low)
            else:
                bearish_divergence = False
            
            if bearish_divergence or close[i] < recent_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish divergence or price breaks above recent high
            lookback = min(10, i)
            recent_high = np.max(high[i-lookback:i+1])
            recent_rsi_high = np.max(rsi[i-lookback:i+1])
            if i >= lookback + 1:
                prev_high = np.max(high[i-lookback-1:i])
                prev_rsi_high = np.max(rsi[i-lookback-1:i])
                bullish_divergence = (high[i] < prev_high) and (rsi[i] > prev_rsi_high)
            else:
                bullish_divergence = False
            
            if bullish_divergence or close[i] > recent_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Look for divergences over lookback period
            lookback = 14
            if i >= lookback:
                # Bullish divergence: price lower low, RSI higher low
                price_lower_low = low[i] < np.min(low[i-lookback:i])
                rsi_higher_low = rsi[i] > np.min(rsi[i-lookback:i])
                bullish_divergence = price_lower_low and rsi_higher_low
                
                # Bearish divergence: price higher high, RSI lower high
                price_higher_high = high[i] > np.max(high[i-lookback:i])
                rsi_lower_high = rsi[i] < np.max(rsi[i-lookback:i])
                bearish_divergence = price_higher_high and rsi_lower_high
                
                # Entry conditions
                if bullish_divergence and vol_confirmed and above_weekly_ema50:
                    position = 1
                    signals[i] = 0.25
                elif bearish_divergence and vol_confirmed and below_weekly_ema50:
                    position = -1
                    signals[i] = -0.25
    
    return signals