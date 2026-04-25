#!/usr/bin/env python3
"""
6h Elder Ray + 12h EMA34 Trend Filter + Volume Spike
Hypothesis: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13. 
Combined with 12h EMA34 trend filter and volume spikes, it captures strong momentum moves 
in both bull and bear markets. Discrete sizing (0.25) controls drawdown. Target: 12-30 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate EMA13 for Elder Ray (on 6h close)
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA13, EMA34_12h, and ATR to propagate
    start_idx = max(13, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema13_val = ema_13[i]
        ema34_12h = ema_34_12h_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Elder Ray components
        bull_power = curr_high - ema13_val   # Bull Power: High - EMA13
        bear_power = curr_low - ema13_val    # Bear Power: Low - EMA13
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND uptrend (price > 12h EMA34) AND volume spike
            long_condition = bull_power > 0 and curr_close > ema34_12h and volume_spike
            # Short: Bear Power < 0 (selling pressure) AND downtrend (price < 12h EMA34) AND volume spike
            short_condition = bear_power < 0 and curr_close < ema34_12h and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or Bear Power turns negative (selling pressure)
            if curr_close <= entry_price - 2.0 * atr_val or bear_power < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or Bull Power turns positive (buying pressure)
            if curr_close >= entry_price + 2.0 * atr_val or bull_power > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0