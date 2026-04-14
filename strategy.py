#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Williams %R and price momentum.
# Williams %R(14) > -20 indicates overbought (short signal), < -80 indicates oversold (long signal).
# Enter only when price is also above/below 6h EMA(20) for momentum confirmation.
# Exit when Williams %R returns to neutral range (-80 to -20) or opposite extreme.
# Uses 1-day Williams %R for higher reliability, avoiding lower timeframe noise.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    # Williams %R(14) on 1d
    williams_len = 14
    if len(df_1d) < williams_len:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest high and lowest low over lookback period
    highest_high = pd.Series(high_1d).rolling(window=williams_len, min_periods=williams_len).max().values
    lowest_low = pd.Series(low_1d).rolling(window=williams_len, min_periods=williams_len).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 6h EMA(20) for momentum confirmation
    ema_len = 20
    if len(close) < ema_len:
        return np.zeros(n)
    
    ema = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(williams_len, ema_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema[i])):
            signals[i] = 0.0
            continue
        
        williams = williams_r_aligned[i]
        price_above_ema = close[i] > ema[i]
        price_below_ema = close[i] < ema[i]
        
        if position == 0:
            # Long when oversold AND price above EMA (bullish momentum)
            if (williams < -80 and 
                price_above_ema):
                position = 1
                signals[i] = position_size
            # Short when overbought AND price below EMA (bearish momentum)
            elif (williams > -20 and 
                  price_below_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when Williams returns to neutral or overbought
            if (williams > -50 or  # returned to neutral or overbought
                williams > -20):   # or reached overbought
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when Williams returns to neutral or oversold
            if (williams < -50 or  # returned to neutral or oversold
                williams < -80):   # or reached oversold
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_WilliamsR_EMA_Momentum_v1"
timeframe = "6h"
leverage = 1.0