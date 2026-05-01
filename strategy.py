#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combo with volume confirmation and chop regime filter.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND volume > 1.5x 20-bar avg AND chop < 61.8 (trending regime).
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND volume confirmation AND chop < 61.8.
# Uses discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Primary timeframe: 12h, HTF: 1d for Elder Ray, 1w for chop filter.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Williams_Alligator_ElderRay_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Elder Ray (EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Load 1w data ONCE before loop for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate chopiness index (14) on 1w
    hl = df_1w['high'].values - df_1w['low'].values
    hc = np.abs(df_1w['high'].values - df_1w['close'].shift(1).values)
    lc = np.abs(df_1w['low'].values - df_1w['close'].shift(1).values)
    tr = np.maximum(hl, np.maximum(hc, lc))
    tr[0] = hl[0]  # first bar
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_1w.sum() / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and NaN
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Calculate Williams Alligator on 12h: SMAs of median price
    median_price = (high + low) / 2
    # Jaws: 13-period SMMA, 8 bars ahead
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 100  # warmup for Alligator and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_13_aligned[i]) or np.isnan(atr[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.5)
        
        # Chop regime filter: trending when chop < 61.8
        trending_regime = chop_aligned[i] < 61.8
        
        # Williams Alligator alignment
        bullish_alligator = jaws[i] < teeth[i] and teeth[i] < lips[i]
        bearish_alligator = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray power
        bull_power = curr_close - ema_13_aligned[i]
        bear_power = ema_13_aligned[i] - curr_close
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator AND bull Elder Power > 0 AND volume confirmation AND trending regime
            if (bullish_alligator and 
                bull_power > 0 and 
                volume_confirm and 
                trending_regime):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator AND bear Elder Power > 0 AND volume confirmation AND trending regime
            elif (bearish_alligator and 
                  bear_power > 0 and 
                  volume_confirm and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator turns bearish OR Elder Power turns negative
            elif (not bullish_alligator or 
                  bull_power <= 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator turns bullish OR Elder Power turns negative
            elif (not bearish_alligator or 
                  bear_power <= 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals