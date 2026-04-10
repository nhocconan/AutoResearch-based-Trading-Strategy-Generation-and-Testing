#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Uses 4h EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Uses daily Camarilla levels (H3/L3) for breakout entries on 1h timeframe
# - Only trade during 08-20 UTC session to avoid low-volume periods
# - Discrete position sizing 0.20 to limit fee churn
# - Target: 15-35 trades/year on 1h timeframe (60-140 total over 4 years)
# - Camarilla pivots provide statistically significant support/resistance levels
# - 4h EMA filter ensures we trade with the higher timeframe trend
# - Session filter reduces noise from Asian session lows

name = "1h_4h_1d_camarilla_ema_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    # H4 = close + 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    # L4 = close - 1.1*(high-low)*1.1/2
    camarilla_range = high_1d - low_1d
    h3 = close_1d + 1.1 * camarilla_range * 1.1 / 4
    l3 = close_1d - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND 4h trend is up (price > EMA)
            if (close[i] > h3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 AND 4h trend is down (price < EMA)
            elif (close[i] < l3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price retreats to the pivot point (close_1d)
            # We'll use a simpler exit: reverse signal or price crosses EMA
            if position == 1:
                # Exit long if price falls below EMA or breaks below L3 (strong reversal)
                if close[i] < ema_50_4h_aligned[i] or close[i] < l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short if price rises above EMA or breaks above H3 (strong reversal)
                if close[i] > ema_50_4h_aligned[i] or close[i] > h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals