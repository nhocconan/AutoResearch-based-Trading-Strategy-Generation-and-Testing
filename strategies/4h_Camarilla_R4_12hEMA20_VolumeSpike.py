#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4 level breakout with 12h EMA20 trend filter and volume spike confirmation.
# Camarilla R4 represents strong resistance in ranging markets; breakout above indicates bullish momentum.
# EMA20 on 12h confirms intermediate trend direction. Volume > 2x average confirms institutional interest.
# Designed for low trade frequency (<50/year) to minimize fee drag in bear markets.
name = "4h_Camarilla_R4_12hEMA20_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA20 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 12h close
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Use previous day's high, low, close for current bar's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # R4 level: Close + (High - Low) * 1.1 / 2
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    # S4 level: Close - (High - Low) * 1.1 / 2
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Need 20 for EMA20 and 1 for rolling
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_20_12h_aligned[i]) or np.isnan(r4[i]) or np.isnan(s4[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_12h = ema_20_12h_aligned[i]
        r4_level = r4[i]
        s4_level = s4[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > R4 AND price > 12h EMA20 (uptrend) AND volume > 2x average
            if close[i] > r4_level and close[i] > ema_12h and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S4 AND price < 12h EMA20 (downtrend) AND volume > 2x average
            elif close[i] < s4_level and close[i] < ema_12h and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S4 OR trend reverses (price < 12h EMA20)
            if close[i] < s4_level or close[i] < ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R4 OR trend reverses (price > 12h EMA20)
            if close[i] > r4_level or close[i] > ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals