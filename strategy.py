#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_RegimeFilter_v1
Hypothesis: Trade daily Camarilla R1/S1 breakouts with weekly EMA34 trend filter and volume confirmation, filtered by choppiness regime (CHOP > 61.8 = range, CHOP < 38.2 = trend). Designed for 1d timeframe to capture multi-day trends with low trade frequency (target: 30-100 trades over 4 years). Weekly HTF trend ensures alignment with major market cycles, reducing whipsaw in bear markets. Volume confirmation and regime filter add robustness. Position size 0.25 balances return and drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get daily data for Camarilla pivot levels (using previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align weekly EMA and daily Camarilla levels to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 1.5x median volume (balanced for 1d frequency)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR for stop (14-period on 1d)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period on 1d) for regime filter
    atr_14_chop = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_chop).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.maximum(max_high_14 - min_low_14, 1e-10)
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop = np.where(chop_denominator > 0, chop_raw, 50.0)
    
    # Price distance from weekly EMA to avoid chop (>0.5%)
    ema_distance = np.abs((close - ema_34_1w_aligned) / ema_34_1w_aligned * 100)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of weekly EMA (34), daily Camarilla (need 2 days), volume median (20), ATR (14), chop (14), distance calc
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema_distance[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1w_val = ema_34_1w_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_14_val = atr_14[i]
        chop_val = chop[i]
        ema_distance_val = ema_distance[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        regime_filter = chop_val < 38.2
        
        if position == 0:
            # Long: break above R1, weekly uptrend (close > weekly EMA34), volume spike, price >0.5% from EMA, trending regime
            long_signal = (high_val > camarilla_r1_val) and \
                          (close_val > ema_34_1w_val) and \
                          (volume_val > 1.5 * vol_median_val) and \
                          (ema_distance_val > 0.5) and \
                          regime_filter
            # Short: break below S1, weekly downtrend (close < weekly EMA34), volume spike, price >0.5% from EMA, trending regime
            short_signal = (low_val < camarilla_s1_val) and \
                           (close_val < ema_34_1w_val) and \
                           (volume_val > 1.5 * vol_median_val) and \
                           (ema_distance_val > 0.5) and \
                           regime_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < weekly EMA34) after minimum holding period
            if bars_since_entry >= 2 and ((low_val < long_stop) or (close_val < ema_34_1w_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > weekly EMA34) after minimum holding period
            if bars_since_entry >= 2 and ((high_val > short_stop) or (close_val > ema_34_1w_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_RegimeFilter_v1"
timeframe = "1d"
leverage = 1.0