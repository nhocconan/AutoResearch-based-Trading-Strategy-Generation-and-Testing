#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 12h trend filter.
# Long when: Alligator jaws (13) < teeth (8) < lips (5) (bullish alignment) AND Elder Bull Power > 0 AND 12h close > 12h EMA34
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND 12h close < 12h EMA34
# Exit when: Alligator alignment breaks OR Elder Power reverses OR 12h trend changes.
# Williams Alligator identifies trend via SMAs (5,8,13). Elder Power measures bull/bear strength relative to EMA13.
# Combines trend confirmation (Alligator) with momentum (Elder Power) and HTF trend filter (12h EMA34).
# Target: 12-37 trades/year on 6h. Discrete sizing 0.25 to balance return and fee drag.
# Works in bull (Alligator bullish + Elder Bull Power > 0) and bear (Alligator bearish + Elder Bear Power < 0).

name = "6h_Alligator_ElderPower_12hEMA34_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for price action and Elder Power calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Alligator and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h: SMAs of median price
    # Median price = (high + low) / 2
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Alligator Lips: SMA(5) of median price
    lips_12h = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values
    # Alligator Teeth: SMA(8) of median price
    teeth_12h = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values
    # Alligator Jaws: SMA(13) of median price
    jaws_12h = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values
    
    # Align Alligator lines to 6h primary timeframe
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws_12h)
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Elder Power on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_6h = pd.Series(df_6h['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_6h = df_6h['high'].values - ema_13_6h
    bear_power_6h = df_6h['low'].values - ema_13_6h
    
    # Align Elder Power to 6h (already on 6h, but align for consistency and proper timing)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for Alligator and EMAs
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaws_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_lips = lips_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_jaws = jaws_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        
        # Alligator alignment conditions
        alligator_bullish = (curr_jaws < curr_teeth) and (curr_teeth < curr_lips)  # Jaws < Teeth < Lips
        alligator_bearish = (curr_jaws > curr_teeth) and (curr_teeth > curr_lips)  # Jaws > Teeth > Lips
        
        # Elder Power conditions
        bull_power_positive = curr_bull_power > 0
        bear_power_negative = curr_bear_power < 0
        
        # 12h trend filter
        uptrend_12h = curr_close > curr_ema_34
        downtrend_12h = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish AND Bull Power > 0 AND 12h uptrend
            if (alligator_bullish and 
                bull_power_positive and 
                uptrend_12h):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0 AND 12h downtrend
            elif (alligator_bearish and 
                  bear_power_negative and 
                  downtrend_12h):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks OR Bull Power <= 0 OR 12h trend turns down
            if (not alligator_bullish or 
                not bull_power_positive or 
                not uptrend_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR Bear Power >= 0 OR 12h trend turns up
            if (not alligator_bearish or 
                not bear_power_negative or 
                not downtrend_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals