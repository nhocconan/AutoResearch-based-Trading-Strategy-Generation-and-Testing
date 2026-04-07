#!/usr/bin/env python3
"""
4h_river_2_bank_v1
Hypothesis: Price moves between support/resistance "banks" (prior session high/low) with mean-reversion tendency. Enter long when price touches prior 12h low with bullish rejection (close > open) and volume > 1.3x average, short when touches prior 12h high with bearish rejection (close < open) and volume > 1.3x average. Uses 12h trend filter (price above/below 50-period EMA) to avoid counter-trend trades. Designed for ~25-40 trades/year to minimize fee drag while capturing mean-reversion in ranging markets and pullbacks in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_river_2_bank_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Calculate 50-period EMA for trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for reference levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not available
        if (np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(open_price[i]) or
            np.isnan(high_12h[-1]) or np.isnan(low_12h[-1])):
            signals[i] = 0.0
            continue
        
        # Get prior completed 12h bar levels
        idx_12h = (i // 48) - 1  # 48 = 4h bars in 12h (shifted by 1 for completed bar)
        if idx_12h < 0:
            signals[i] = 0.0
            continue
            
        prior_12h_high = high_12h[idx_12h]
        prior_12h_low = low_12h[idx_12h]
        
        # Volume confirmation: > 1.3x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.3)
        
        # Candlestick rejection: bullish = close > open, bearish = close < open
        bullish_rejection = close[i] > open_price[i]
        bearish_rejection = close[i] < open_price[i]
        
        if position == 1:  # Long position
            # Exit: price closes below prior 12h low (break of support)
            if close[i] < prior_12h_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above prior 12h high (break of resistance)
            if close[i] > prior_12h_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: touch prior 12h low with bullish rejection + price > 12h EMA50
                if (low[i] <= prior_12h_low * 1.001 and  # Allow small tolerance for touch
                    bullish_rejection and 
                    close[i] > ema_50_12h_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: touch prior 12h high with bearish rejection + price < 12h EMA50
                elif (high[i] >= prior_12h_high * 0.999 and  # Allow small tolerance for touch
                      bearish_rejection and 
                      close[i] < ema_50_12h_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals