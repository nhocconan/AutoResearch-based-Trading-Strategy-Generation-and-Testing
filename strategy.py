#!/usr/bin/env python3
"""
4h_ema_trend_reversal_v2
Hypothesis: On 4h timeframe, use EMA trend filter with RSI divergence and volume confirmation.
In bull markets: enter long when EMA20 > EMA50, RSI shows bullish divergence (higher low in RSI while price makes lower low), and volume > 1.3x average.
In bear markets: enter short when EMA20 < EMA50, RSI shows bearish divergence (lower high in RSI while price makes higher high), and volume > 1.3x average.
Exit when EMA crossover reverses or RSI reaches extreme levels (70 for long, 30 for short).
This strategy captures trend reversals with confirmation, works in both bull/bear via EMA filter, and limits trades via strict divergence conditions.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_trend_reversal_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMAs for trend filter
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if EMA20 crosses below EMA50 (trend reversal)
            if ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1]:
                exit_long = True
            # Exit if RSI reaches overbought (70)
            elif rsi[i] >= 70:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if EMA20 crosses above EMA50 (trend reversal)
            if ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1]:
                exit_short = True
            # Exit if RSI reaches oversold (30)
            elif rsi[i] <= 30:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need at least 3 periods for divergence check
            if i < 3:
                signals[i] = 0.0
                continue
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = False
            if (low[i] < low[i-1] and low[i-1] < low[i-2] and  # price lower low
                rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]):   # RSI higher low
                bull_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = False
            if (high[i] > high[i-1] and high[i-1] > high[i-2] and  # price higher high
                rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]):      # RSI lower high
                bear_div = True
            
            # Long entry: EMA20 > EMA50, bullish divergence, volume confirmation
            long_entry = False
            if (ema20[i] > ema50[i] and bull_div and vol_confirm):
                long_entry = True
            
            # Short entry: EMA20 < EMA50, bearish divergence, volume confirmation
            short_entry = False
            if (ema20[i] < ema50[i] and bear_div and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals