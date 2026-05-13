# 1d_Camarilla_R1_S1_Breakout_1wTrend
# Hypothesis: On 1d timeframe, price breaking above Camarilla R1 or below S1 with weekly trend confirmation (close above/below 20-week EMA) and volume surge (1.5x 20-day average) captures institutional breakout moves. Works in bull/bear by following weekly trend, avoiding counter-trend trades. Low frequency: ~10-20 trades/year per symbol.

#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data (same as primary for Camarilla calculation)
    df_1d = prices.copy()
    
    # Load 1W data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week EMA for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-calculate daily typical price for Camarilla
    typical_price = (high + low + close) / 3.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Camarilla
        if np.isnan(ema20_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Camarilla levels from previous day (i-1)
        if i == 0:
            signals[i] = 0.0
            continue
            
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        range_ = ph - pl
        if range_ <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        r1 = pc + 1.1 * range_ / 12
        s1 = pc - 1.1 * range_ / 12
        
        # Volume confirmation: current volume > 1.5x 20-day average
        if i >= 20:
            vol_avg = np.mean(volume[i-20:i])
            vol_surge = volume[i] > 1.5 * vol_avg
        else:
            vol_surge = False
        
        # Weekly trend filter
        price_above_weekly_ema = close[i] > ema20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # LONG: Break above R1 + weekly uptrend + volume surge
            if close[i] > r1 and price_above_weekly_ema and vol_surge:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 + weekly downtrend + volume surge
            elif close[i] < s1 and price_below_weekly_ema and vol_surge:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below R1 or weekly trend turns down
            if close[i] < r1 or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above S1 or weekly trend turns up
            if close[i] > s1 or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data (same as primary for Camarilla calculation)
    df_1d = prices.copy()
    
    # Load 1W data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week EMA for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-calculate daily typical price for Camarilla
    typical_price = (high + low + close) / 3.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Camarilla
        if np.isnan(ema20_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Camarilla levels from previous day (i-1)
        if i == 0:
            signals[i] = 0.0
            continue
            
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        range_ = ph - pl
        if range_ <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        r1 = pc + 1.1 * range_ / 12
        s1 = pc - 1.1 * range_ / 12
        
        # Volume confirmation: current volume > 1.5x 20-day average
        if i >= 20:
            vol_avg = np.mean(volume[i-20:i])
            vol_surge = volume[i] > 1.5 * vol_avg
        else:
            vol_surge = False
        
        # Weekly trend filter
        price_above_weekly_ema = close[i] > ema20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # LONG: Break above R1 + weekly uptrend + volume surge
            if close[i] > r1 and price_above_weekly_ema and vol_surge:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 + weekly downtrend + volume surge
            elif close[i] < s1 and price_below_weekly_ema and vol_surge:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below R1 or weekly trend turns down
            if close[i] < r1 or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above S1 or weekly trend turns up
            if close[i] > s1 or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals