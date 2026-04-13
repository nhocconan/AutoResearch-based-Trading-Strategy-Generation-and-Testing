#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h primary with 1d HTF - Camarilla pivot breakout with volume confirmation and chop regime filter
    # Designed to capture institutional volume-driven breakouts from key daily pivot levels in both bull and bear markets
    # Uses Camarilla levels (H3/L3) as breakout triggers, volume > 2x average for confirmation, and chop < 61.8 for trending regime
    # Target: 50-150 total trades over 4 years (12-37/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Calculate 1d Chopiness Index (14-period) for regime filter
    def calculate_chop(high, low, close, window=14):
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=1).sum()
        hh = pd.Series(high).rolling(window=window, min_periods=1).max()
        ll = pd.Series(low).rolling(window=window, min_periods=1).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
        return chop.values
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    
    # Calculate 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_avg_20[i]
        
        # Regime filter: chop < 61.8 (trending market)
        trending_regime = chop_1d_aligned[i] < 61.8
        
        # Trend filter: price above/below 1w EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions at Camarilla H3/L3 levels
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions: breakout + volume + regime + trend alignment
        enter_long = breakout_up and volume_confirmed and trending_regime and uptrend
        enter_short = breakout_down and volume_confirmed and trending_regime and downtrend
        
        # Exit conditions: price returns to opposite Camarilla level or midpoint
        midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
        exit_long = position == 1 and close[i] <= midpoint
        exit_short = position == -1 and close[i] >= midpoint
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0