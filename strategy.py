#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter, volume spike (>2x average), and choppiness regime filter (CHOP > 61.8 = range). 
In range markets: price breaks above R1 with 1d uptrend and high volume → long; breaks below S1 with 1d downtrend and high volume → short. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 trades over 4 years (19-50/year) on 4h timeframe.
Requires BTC/ETH edge via 1d trend, volume, and regime filters; avoids SOL-only bias by requiring multi-factor confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and CHOP
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) on 4h data for regime filter
    def calculate_chop(high, low, close, window=14):
        """Calculate Choppiness Index: higher values = more choppy/ranging market"""
        atr = []
        for i in range(len(high)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr.append(tr)
        
        atr_series = pd.Series(atr)
        tr_sum = atr_series.rolling(window=window, min_periods=window).sum()
        hh = pd.Series(high).rolling(window=window, min_periods=window).max()
        ll = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(window)
        return chop.values
    
    chop = calculate_chop(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 1 for Camarilla, 34 for EMA, 20 for volume, 14 for CHOP)
    start_idx = max(1, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using previous day's OHLC
        # For 4h timeframe, previous day = previous 6 bars
        prev_1d_idx = i - 6
        if prev_1d_idx < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        prev_high = high[prev_1d_idx]
        prev_low = low[prev_1d_idx]
        prev_close = close[prev_1d_idx]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R1 and S1 levels
        R1 = prev_close + (range_val * 1.1 / 12)
        S1 = prev_close - (range_val * 1.1 / 12)
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        chop_val = chop[i]
        
        # Skip if any data not ready
        if np.isnan(R1) or np.isnan(S1) or np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(chop_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        regime_filter = chop_val > 61.8
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above R1 with 1d uptrend, volume confirmation, and in ranging market
        long_condition = (close_val > R1) and (close_val > ema_val) and volume_confirmed and regime_filter
        # Short logic: price breaks below S1 with 1d downtrend, volume confirmation, and in ranging market
        short_condition = (close_val < S1) and (close_val < ema_val) and volume_confirmed and regime_filter
        
        # Exit logic: trend reversal, opposite breakout, or regime change to trending
        exit_long = (close_val < ema_val) or (close_val < S1) or (chop_val <= 61.8)
        exit_short = (close_val > ema_val) or (close_val > S1) or (chop_val <= 61.8)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0