#!/usr/bin/env python3
name = "6h_1d_1w_RangeBreakout_Pullback_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily ATR for volatility filter
    atr_period = 14
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Weekly trend: EMA(21) on weekly close
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily range breakout: Donchian(10) on daily
    donch_high_1d = pd.Series(df_1d['high']).rolling(window=10, min_periods=10).max().values
    donch_low_1d = pd.Series(df_1d['low']).rolling(window=10, min_periods=10).min().values
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # 6h volume spike detection
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 10)  # Wait for ATR and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or 
            np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when volatility is sufficient (avoid choppy markets)
        vol_condition = atr_1d_aligned[i] > np.nanpercentile(atr_1d_aligned[:i+1], 50)
        
        if position == 0:
            # Long: breakout above daily Donchian high with volume in weekly uptrend
            long_breakout = close[i] > donch_high_1d_aligned[i]
            long_volume = volume[i] > vol_ma_10[i] * 1.5
            weekly_uptrend = ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]
            
            if long_breakout and long_volume and weekly_uptrend and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below daily Donchian low with volume in weekly downtrend
            elif close[i] < donch_low_1d_aligned[i] and volume[i] > vol_ma_10[i] * 1.5 and \
                 ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: pullback to 50% of daily range or volume drops
            daily_range = donch_high_1d_aligned[i] - donch_low_1d_aligned[i]
            pullback_level = donch_low_1d_aligned[i] + (daily_range * 0.5)
            if close[i] < pullback_level or volume[i] < vol_ma_10[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: pullback to 50% of daily range or volume drops
            daily_range = donch_high_1d_aligned[i] - donch_low_1d_aligned[i]
            pullback_level = donch_high_1d_aligned[i] - (daily_range * 0.5)
            if close[i] > pullback_level or volume[i] < vol_ma_10[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s range breakout with pullback entry using daily Donchian channels
# - Uses daily Donchian(10) breakouts for directional entries
# - Filters by weekly EMA(21) trend to avoid counter-trend trades
# - Requires volume spike (1.5x) and above-median volatility to avoid chop
# - Exits on 50% retracement of the daily range or volume decline
# - Designed to work in both bull and bear markets via weekly trend filter
# - Targets 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag
# - Weekly trend filter ensures we only trade in the direction of higher timeframe momentum
# - Pullback exit captures mean reversion within the trend while allowing trends to run