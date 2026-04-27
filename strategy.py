#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_Regime
Hypothesis: 4h strategy using Donchian(20) breakouts with 12h EMA50 trend filter and volume confirmation, plus choppiness regime filter. Donchian breakouts capture strong momentum moves, 12h EMA50 ensures alignment with intermediate trend, volume confirmation filters weak breakouts, and choppiness regime filter avoids whipsaws in sideways markets. Designed for BTC/ETH robustness. Targets 75-200 trades over 4 years (19-50/year) with 0.25 position size.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 1d data for choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_period = 14
    chop_period = 14
    
    # Calculate ATR for chop
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate highest high and lowest low over chop_period
    highest_high = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Chop = 100 * log10(sum(TR over chop_period) / log10(highest_high - lowest_low)) / log10(chop_period)
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    range_hl = highest_high - lowest_low
    chop = 100 * np.log10(sum_tr) / np.log10(range_hl) / np.log10(chop_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)  # CHOP uses completed 1d bar
    
    # Donchian channels (20-period) on 4h data
    donch_period = 20
    donch_high = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    donch_low = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 12h EMA50 (50), 1d CHOP (14+14), Donchian (20), vol avg (20)
    start_idx = max(50, 14 + 14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema_val = ema_50_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 12h EMA50 alignment, volume confirmation, and trending regime (CHOP < 38.2)
            long_condition = (close_val > donch_high_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            chop_val < 38.2)
            short_condition = (close_val < donch_low_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             chop_val < 38.2)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low (trailing stop) OR 12h EMA50 crosses below price (trend change)
            if close_val < donch_low_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian high (trailing stop) OR 12h EMA50 crosses above price (trend change)
            if close_val > donch_high_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0