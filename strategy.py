#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
    # Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
    # Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw
    # Volume confirmation: volume > 2.0 * 20-period average to avoid false signals
    # Chop regime: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion at extremes
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Alligator on 1d
    # Median price = (high + low) / 2
    median_price_1d = (high_1d + low_1d) / 2
    
    # Jaw: 13-period SMA, smoothed by 8 periods
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean()
    jaw_1d = jaw_1d.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean()
    teeth_1d = teeth_1d.rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean()
    lips_1d = lips_1d.rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate Chopiness Index on 1d (14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # True Range with proper handling
        tr[0] = high[0] - low[0]
        atr_sum = np.zeros(len(close))
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        atr = np.where(np.arange(len(close)) >= period-1, atr_sum[period-1:] / period, np.nan)
        atr_full = np.full(len(close), np.nan)
        atr_full[period-1:] = atr
        
        # Chop calculation
        max_high = np.full(len(close), np.nan)
        min_low = np.full(len(close), np.nan)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        chop = np.full(len(close), np.nan)
        for i in range(period-1, len(close)):
            if atr_full[i] > 0 and (max_high[i] - min_low[i]) > 0:
                log_sum = np.log10(atr_full[i] * period) / np.log10(2)
                log_range = np.log10(max_high[i] - min_low[i]) / np.log10(2)
                chop[i] = 100 * log_sum / log_range
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        in_chop_regime = chop_aligned[i] > 61.8
        
        # Entry logic: Alligator alignment with volume and chop filter
        long_entry = False
        short_entry = False
        
        if bullish_alignment and in_chop_regime:
            long_entry = volume_spike[i]
        elif bearish_alignment and in_chop_regime:
            short_entry = volume_spike[i]
        
        # Exit logic: opposite alignment or chop regime breakdown
        long_exit = (not bullish_alignment) or (not in_chop_regime)
        short_exit = (not bearish_alignment) or (not in_chop_regime)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_alligator_chop_volume_v1"
timeframe = "12h"
leverage = 1.0