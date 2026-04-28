#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter.
# Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (all lines intertwined).
# Elder Ray (Bull/Bear Power) measures trend strength via EMA13 deviation.
# In choppy markets (Alligator sleeping): fade extreme Elder Ray readings.
# In trending markets (Alligator awakening): follow Elder Ray momentum.
# Uses 1d EMA34 for regime filter: bull/bear defined by price vs EMA34.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels minimize fee churn. Targets 12-37 trades/year on 6h.

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for regime filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: SMAs of median price (typical price)
    typical_price = (high + low + close) / 3.0
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to LTF (no extra delay needed for SMAs)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw) if len(df_1d) >= 13 else np.full(n, np.nan)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth) if len(df_1d) >= 8 else np.full(n, np.nan)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips) if len(df_1d) >= 5 else np.full(n, np.nan)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13  # negative values indicate bearish pressure
    
    # Regime detection: Alligator sleeping (chop) vs awakening (trend)
    # Sleeping: all lines within 1% of each other (market choppy)
    jaw_teeth_diff = np.abs(jaw_aligned - teeth_aligned) / np.maximum(np.abs(jaw_aligned), 1e-10)
    teeth_lips_diff = np.abs(teeth_aligned - lips_aligned) / np.maximum(np.abs(teeth_aligned), 1e-10)
    lips_jaw_diff = np.abs(lips_aligned - jaw_aligned) / np.maximum(np.abs(lips_aligned), 1e-10)
    max_diff = np.maximum(jaw_teeth_diff, np.maximum(teeth_lips_diff, lips_jaw_diff))
    alligator_sleeping = max_diff < 0.01  # choppy regime
    alligator_awakening = max_diff >= 0.01  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Alligator state
        sleeping = alligator_sleeping[i]
        awakening = alligator_awakening[i]
        
        # Elder Ray extremes (normalized by price for comparability)
        bull_extreme = bull_power[i] > (close[i] * 0.02)  # strong bullish pressure
        bear_extreme = bear_power[i] < -(close[i] * 0.02)  # strong bearish pressure
        
        # Logic: 
        # In chop (sleeping): fade extreme Elder Ray readings (mean reversion)
        # In trend (awakening): follow Elder Ray momentum with 1d EMA34 filter
        long_entry = False
        short_entry = False
        
        if sleeping:
            # Choppy market: fade extremes
            long_entry = bear_extreme and price_above_ema  # long on bear exhaustion in uptrend
            short_entry = bull_extreme and price_below_ema  # short on bull exhaustion in downtrend
        else:
            # Trending market: follow momentum with trend filter
            long_entry = bull_extreme and price_above_ema
            short_entry = bear_extreme and price_below_ema
        
        # Exit conditions: opposite Elder Ray extreme or regime change
        long_exit = bull_extreme or (sleeping and not awakening)  # exit long on bull extreme or regime to chop
        short_exit = bear_extreme or (sleeping and not awakening)  # exit short on bear extreme or regime to chop
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals