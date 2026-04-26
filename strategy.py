#!/usr/bin/env python3
"""
6h_ADX_DMI_Trend_12h_EMA34_Filter_v1
Hypothesis: Use 6h timeframe with ADX(14) > 25 and +DI > -DI for uptrend, -DI > +DI for downtrend, confirmed by 12h EMA34 trend filter. Targets 12-37 trades/year to minimize fee drag. Works in trending markets (both bull/bear) by using ADX for trend strength and direction, with 12h EMA34 as higher-timeframe confirmation. Includes ATR-based stoploss to manage risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX and DMI (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 14*2 for ADX/DMI, 34 for 12h EMA
    start_idx = max(28, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for trend entry with ADX strength and 12h EMA confirmation
            # Long: ADX > 25, +DI > -DI, and 12h EMA34 uptrend
            long_entry = (adx[i] > 25) and (plus_di[i] > minus_di[i]) and \
                       (i > start_idx and ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1])
            # Short: ADX > 25, -DI > +DI, and 12h EMA34 downtrend
            short_entry = (adx[i] > 25) and (minus_di[i] > plus_di[i]) and \
                        (i > start_idx and ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1])
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend weakening or ATR stoploss
            exit_condition = (adx[i] < 20) or (plus_di[i] < minus_di[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend weakening or ATR stoploss
            exit_condition = (adx[i] < 20) or (minus_di[i] < plus_di[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_DMI_Trend_12h_EMA34_Filter_v1"
timeframe = "6h"
leverage = 1.0