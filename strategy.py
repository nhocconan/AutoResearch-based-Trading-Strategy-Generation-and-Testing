#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 trend + volume confirmation
# Long when Bull Power > 0 (price > EMA13) and Bear Power < 0 (EMA13 > price) with confirmation
# Actually: Elder Ray = Bull Power = Close - EMA13, Bear Power = EMA13 - Close
# We use: Bull Power > 0 AND Bear Power < 0 is impossible. Instead:
# Long when Bull Power > 0 AND Bear Power declining (momentum)
# Short when Bear Power > 0 AND Bull Power declining
# Trend filter: price > 1d EMA34 for long, price < 1d EMA34 for short
# Volume: current 6h volume > 1.5x 20-period EMA of 6h volume
# Designed for 6h timeframe to target 15-35 trades/year (60-140 total over 4 years)
# Elder Ray measures bull/bear power relative to EMA; trend filter avoids counter-trend
# Volume ensures institutional participation

name = "6h_ElderRay_Energy_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    
    # Calculate Elder Ray components (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13  # Bull Power = Close - EMA13
    bear_power = ema_13 - close  # Bear Power = EMA13 - Close
    
    # Calculate 6h volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13)  # warmup for EMA34 and EMA13
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period EMA
        vol_filter = volume[i] > 1.5 * vol_ema_20[i]
        
        if position == 0:
            # Look for entry: Elder Ray momentum + trend + volume
            # Long: Bull Power > 0 AND Bear Power declining (more negative) AND price > EMA34
            # Short: Bear Power > 0 AND Bull Power declining (more negative) AND price < EMA34
            bull_power_rising = bull_power[i] > bull_power[i-1]
            bear_power_rising = bear_power[i] > bear_power[i-1]
            
            long_condition = bull_power[i] > 0 and bear_power_rising == False and close[i] > ema_34_aligned[i] and vol_filter
            short_condition = bear_power[i] > 0 and bull_power_rising == False and close[i] < ema_34_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power becomes positive (bears taking over) OR price < EMA34
            if bear_power[i] > 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power becomes positive (bulls taking over) OR price > EMA34
            if bull_power[i] > 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals