#!/usr/bin/env python3
"""
1d_WeeklyEMA34_CamarillaPivotBreakout_v1
Hypothesis: Daily Camarilla pivot (R1/S1) breakouts in direction of weekly trend (price > weekly EMA34) with volume confirmation (>1.5x 20-day MA). 
In bull weekly regime (price > weekly EMA34), take longs on R1 breakouts; in bear weekly regime (price < weekly EMA34), take shorts on S1 breakouts. 
Weekly EMA34 provides stable trend filter; Camarilla pivots capture intraday reversal points; volume filter ensures participation. 
Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year) by requiring confluence of breakout, weekly trend, and volume.
Stoploss: exit when price crosses weekly EMA34 (trend reversal). Discrete sizing (0.25) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily typical price for Camarilla calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    typical_price = (high + low + close) / 3.0
    
    # Previous day's typical price (for pivot calculation)
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan
    
    # Camarilla pivot levels (based on previous day)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12.0
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12.0
    
    # === Daily volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # start after warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        r1_val = camarilla_r1[i]
        s1_val = camarilla_s1[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly trend regime
        is_bull = price > ema_34_1w_val
        is_bear = price < ema_34_1w_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long R1 breakout
                long_condition = (price > r1_val) and vol_conf
                short_condition = False  # no shorts in bull regime
            else:  # bear regime
                # Bear regime: short S1 breakdown
                short_condition = (price < s1_val) and vol_conf
                long_condition = False  # no longs in bear regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit when price crosses weekly EMA34 (trend reversal)
            if position == 1 and price < ema_34_1w_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > ema_34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA34_CamarillaPivotBreakout_v1"
timeframe = "1d"
leverage = 1.0