#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h WilliamsVix fix + 1w Supertrend + volume spike
    # WilliamsVix < -80 oversold (long), > -20 overbought (short) in ranging markets (CHOP > 61.8)
    # 1w Supertrend(10,3.0) for trend filter to avoid counter-trend
    # Volume > 2.0x 24-period MA confirms momentum breakout
    # Discrete position sizing 0.25 to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend(10,3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(high_1w) - pd.Series(low_1w).shift(1)
    tr3 = pd.Series(low_1w).shift(1) - pd.Series(close_1w).shift(1)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(len(close_1w), np.nan)
    direction = np.full(len(close_1w), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, len(close_1w)):
        if i == atr_period:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upper_band[i-1]:
                supertrend[i] = lower_band[i] if close_1w[i] > upper_band[i-1] else upper_band[i]
                direction[i] = -1 if close_1w[i] > upper_band[i-1] else 1
            else:
                supertrend[i] = upper_band[i] if close_1w[i] < lower_band[i-1] else lower_band[i]
                direction[i] = 1 if close_1w[i] < lower_band[i-1] else -1
    
    # Align Supertrend direction to 12h
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # WilliamsVix (Williams %R + Bollinger Bands %) on 12h
    # Williams %R (14-period)
    highest_14 = np.full(n, np.nan)
    lowest_14 = np.full(n, np.nan)
    
    for i in range(14, n):
        highest_14[i] = np.max(high[i-14:i])
        lowest_14[i] = np.min(low[i-14:i])
    
    williams_r = np.full(n, np.nan)
    for i in range(14, n):
        if highest_14[i] != lowest_14[i]:
            williams_r[i] = (highest_14[i] - close[i]) / (highest_14[i] - lowest_14[i]) * -100
        else:
            williams_r[i] = -50.0
    
    # Bollinger Bands %B (20,2)
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std * std_bb)
    lower_bb = sma_bb - (bb_std * std_bb)
    bb_percent_b = np.full(n, np.nan)
    for i in range(bb_period, n):
        if upper_bb[i] != lower_bb[i]:
            bb_percent_b[i] = (close[i] - lower_bb[i]) / (upper_bb[i] - lower_bb[i])
        else:
            bb_percent_b[i] = 0.5
    
    # WilliamsVix = Williams %R * (1 - |%B - 0.5| * 2)
    williams_vix = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(williams_r[i]) and not np.isnan(bb_percent_b[i]):
            deviation = abs(bb_percent_b[i] - 0.5) * 2  # 0 at middle, 1 at bands
            williams_vix[i] = williams_r[i] * (1 - deviation)
        else:
            williams_vix[i] = np.nan
    
    # Choppiness Index (14) for regime filter
    chop_period = 14
    atr_12h = np.full(n, np.nan)
    tr_12h = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_12h[chop_period:] = pd.Series(tr_12h).rolling(window=chop_period, min_periods=chop_period).mean().values[chop_period:]
    
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    chop = np.full(n, 50.0)  # default to neutral
    for i in range(chop_period, n):
        if atr_12h[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(sum(atr_12h[i-chop_period+1:i+1]) / np.log10(chop_period) / (max_high[i] - min_low[i])) / np.log10(chop_period)
        else:
            chop[i] = 50.0
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(24, n):
        if vol_ma_24[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_24[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_vix[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w Supertrend direction
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # WilliamsVix mean reversion conditions in ranging market (CHOP > 61.8)
        oversold = williams_vix[i] < -80
        overbought = williams_vix[i] > -20
        ranging_market = chop[i] > 61.8
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = oversold and (vol_ratio[i] > 2.0) and ranging_market and uptrend
        short_entry = overbought and (vol_ratio[i] > 2.0) and ranging_market and downtrend
        
        # Exit conditions: WilliamsVix returns to midpoint (-50) or regime changes
        long_exit = williams_vix[i] > -50 or chop[i] < 38.2
        short_exit = williams_vix[i] < -50 or chop[i] < 38.2
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_williamsvix_supertrend_chop_vol_v1"
timeframe = "12h"
leverage = 1.0