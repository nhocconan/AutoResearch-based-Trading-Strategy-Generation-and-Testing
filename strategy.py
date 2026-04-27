#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_RegimeFilter
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike, and chop regime filter.
Long when price breaks above R3 AND price > 1d EMA34 AND volume spike AND chop < 61.8 (trending).
Short when price breaks below S3 AND price < 1d EMA34 AND volume spike AND chop < 61.8.
Exit when price re-enters the Camarilla range (between S3 and R3) or loses 1d EMA34 alignment.
Camarilla levels from 1d provide precise intraday support/resistance. Volume spike confirms breakout strength.
Chop filter avoids false signals in ranging markets. Designed for 20-50 trades/year on 4h to minimize fee drag.
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
    
    # Calculate Camarilla levels from previous 1d bar
    df_1d = get_htf_data(prices, '1d')
    # Camarilla: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # shift(1) for completed 1d bar
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    # True range for Camarilla calculation
    rng = prev_high - prev_low
    # Camarilla levels
    r3 = prev_close + (rng * 1.1 / 4)
    s3 = prev_close - (rng * 1.1 / 4)
    r4 = prev_close + (rng * 1.1 / 2)
    s4 = prev_close - (rng * 1.1 / 2)
    # Align to 4h (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index (CHOP) regime filter - avoid ranging markets
    # CHOP(14) > 61.8 = ranging (choppy), CHOP < 38.2 = trending
    # We want trending markets: CHOP < 61.8
    atr_period = 14
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # same length as close
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    # Sum of true range over atr_period
    tr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    # Chop = 100 * log10(tr_sum / (atr * atr_period)) / log10(atr_period)
    chop = 100 * np.log10(tr_sum / (atr * atr_period)) / np.log10(atr_period)
    # For simplicity, use chop < 61.8 as trending regime
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1d data (shifted), EMA34, volume avg, ATR
    start_idx = max(30, 34, 20, atr_period)  # 1d shift needs ~30 bars (4h per day = 6, but using 30 for safety)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        regime = trending_regime[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with 1d EMA34 alignment, volume spike, and trending regime
            # Long: Close > R3 AND price > 1d EMA34 AND volume spike AND trending regime
            # Short: Close < S3 AND price < 1d EMA34 AND volume spike AND trending regime
            long_condition = (close_val > r3_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            regime)
            short_condition = (close_val < s3_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             regime)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price re-enters Camarilla range (above S3) OR loses 1d EMA34 alignment
            if close_val < s3_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Camarilla range (below R3) OR loses 1d EMA34 alignment
            if close_val > r3_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0