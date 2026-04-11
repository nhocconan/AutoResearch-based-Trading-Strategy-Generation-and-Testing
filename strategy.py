#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Long when price breaks above 4h Camarilla H3 level + 1d volume > 1.8x 20-period average + chop < 61.8 (trending)
# - Short when price breaks below 4h Camarilla L3 level + 1d volume > 1.8x 20-period average + chop < 61.8 (trending)
# - Exit when price reverts to 4h Camarilla pivot point or ATR stoploss triggered
# - Uses ATR-based stoploss (signal→0 when adverse move > 2.5*ATR)
# - Target: 15-35 trades/year to minimize fee drag while capturing strong institutional breakouts
# - Works in bull/bear: Camarilla levels identify key support/resistance; volume confirms institutional participation; chop filter avoids whipsaws

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 4h data ONCE before loop for Camarilla calculations and ATR (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for volume and chop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 4h Camarilla levels (based on previous 4h bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Calculate previous bar's OHLC for Camarilla (shift by 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_open = np.roll(open_4h, 1)
    
    # First bar has no previous data
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    prev_open[0] = open_4h[0]
    
    # Camarilla levels calculation
    rang = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_h3 = camarilla_pivot + (rang * 1.1 / 4.0)
    camarilla_l3 = camarilla_pivot - (rang * 1.1 / 4.0)
    camarilla_h4 = camarilla_pivot + (rang * 1.1 / 2.0)
    camarilla_l4 = camarilla_pivot - (rang * 1.1 / 2.0)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # Pre-compute 4h ATR(20) for stoploss
    tr1 = pd.Series(high_4h).rolling(2).max() - pd.Series(low_4h).rolling(2).min()
    tr2 = abs(pd.Series(high_4h).shift(1) - pd.Series(close_4h))
    tr3 = abs(pd.Series(low_4h).shift(1) - pd.Series(close_4h))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_4h, atr_20)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d chop regime filter (Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = pd.Series(high_1d).rolling(2).max() - pd.Series(low_1d).rolling(2).min()
    tr2_1d = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d))
    tr3_1d = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero/negative
    denominator = hh_14 - ll_14
    chop_raw = np.where((denominator > 0) & (sum_tr_14 > 0), 
                        100 * np.log10(sum_tr_14 / denominator) / np.log10(14), 
                        50.0)  # Default to neutral when invalid
    chop = chop_raw
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(atr_20_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h bar data
        close_4h_current = close[i]
        high_4h_current = high[i]
        low_4h_current = low[i]
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume SMA (tighter threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: chop < 61.8 indicates trending market (good for breakouts)
        chop_current = chop_aligned[i]
        trending_regime = chop_current < 61.8
        
        # Camarilla breakout conditions
        breakout_up = close_4h_current > camarilla_h3_aligned[i]
        breakout_down = close_4h_current < camarilla_l3_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and vol_confirm and trending_regime
        enter_short = breakout_down and vol_confirm and trending_regime
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (close_4h_current < camarilla_pivot_aligned[i] or  # Revert to pivot
                     close_4h_current < entry_price - 2.5 * atr_20_aligned[i]))  # ATR stoploss
        exit_short = (position == -1 and 
                     (close_4h_current > camarilla_pivot_aligned[i] or  # Revert to pivot
                      close_4h_current > entry_price + 2.5 * atr_20_aligned[i]))  # ATR stoploss
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_4h_current
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_4h_current
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals