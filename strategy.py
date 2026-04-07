#!/usr/bin/env python3
"""
4h_dema_crossover_12h_trend_volume_v1
Hypothesis: On 4h timeframe, use 12h DEMA crossover (fast/slow) for trend direction and strength, with volume confirmation to filter false signals. Enter long when fast DEMA crosses above slow DEMA with price above both and volume above average; enter short when fast DEMA crosses below slow DEMA with price below both and volume above average. Exit when crossover reverses. DEMA reduces lag vs EMA while maintaining smoothness, capturing trends with fewer whipsaws. Volume confirmation ensures institutional participation, reducing false breakouts. Works in bull/bear via trend filter and momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_dema_crossover_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for DEMA crossover
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate DEMA (Double EMA) on 12h data
    # DEMA = 2*EMA - EMA(EMA)
    ema1 = pd.Series(close_12h).ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    dema_fast = (2 * ema1 - ema2).values
    
    ema1_slow = pd.Series(close_12h).ewm(span=50, adjust=False).mean()
    ema2_slow = ema1_slow.ewm(span=50, adjust=False).mean()
    dema_slow = (2 * ema1_slow - ema2_slow).values
    
    # Align indicators to 4h timeframe
    dema_fast_4h = align_htf_to_ltf(prices, df_12h, dema_fast)
    dema_slow_4h = align_htf_to_ltf(prices, df_12h, dema_slow)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(dema_fast_4h[i]) or np.isnan(dema_slow_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend direction from DEMA crossover
        bullish = dema_fast_4h[i] > dema_slow_4h[i]
        bearish = dema_fast_4h[i] < dema_slow_4h[i]
        
        # Price position relative to DEMAs
        price_above = close[i] > dema_fast_4h[i] and close[i] > dema_slow_4h[i]
        price_below = close[i] < dema_fast_4h[i] and close[i] < dema_slow_4h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if DEMA crossover turns bearish
            if bearish:
                exit_long = True
            # Exit if price falls below fast DEMA
            elif close[i] < dema_fast_4h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if DEMA crossover turns bullish
            if bullish:
                exit_short = True
            # Exit if price rises above fast DEMA
            elif close[i] > dema_fast_4h[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Bullish crossover with price above both DEMAs and volume confirmation
            if bullish and dema_fast_4h[i-1] <= dema_slow_4h[i-1]:
                if price_above and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Bearish crossover with price below both DEMAs and volume confirmation
            if bearish and dema_fast_4h[i-1] >= dema_slow_4h[i-1]:
                if price_below and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals