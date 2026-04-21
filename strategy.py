#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Chop Regime
# Long when Alligator jaws (SMMA13) < teeth (SMMA8) < lips (SMMA5) and price > lips
# Short when jaws > teeth > lips and price < lips
# Requires 1d volume > 2.0x 20-day average and 12h chop index < 38.2 (trending)
# Williams Alligator uses smoothed moving averages (SMMA) to identify trends
# Works in both bull (strong alignment up) and bear (strong alignment down)
# Volume confirms conviction, chop filter avoids ranging markets
# Target: 15-25 trades/year by requiring multiple confluence factors

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d SMMA for Williams Alligator (Smoothed Moving Average)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # SMMA calculation (similar to EMA but with different smoothing)
    def smma(data, period):
        sma = np.full_like(data, np.nan)
        smma = np.full_like(data, np.nan)
        sma[period-1] = np.mean(data[:period])
        smma[period-1] = sma[period-1]
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (i-1) + data[i]) / i
            smma[i] = (smma[i-1] * (period-1) + data[i]) / period
        return smma
    
    # Alligator lines: Lips (SMMA5), Teeth (SMMA8), Jaw (SMMA13)
    lips = smma(close_1d, 5)
    teeth = smma(close_1d, 8)
    jaw = smma(close_1d, 13)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Chop Index (trend strength indicator)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = np.full(n, np.nan)
    atr[13] = np.mean(tr[:14])
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr = np.full(n, np.nan)
    for i in range(13, n):
        if i == 13:
            sum_atr[i] = np.sum(atr[max(0, i-13):i+1])
        else:
            sum_atr[i] = sum_atr[i-1] - atr[i-14] + atr[i]
    
    # Chop Index = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            max_high[i] = high[i]
            min_low[i] = low[i]
        else:
            max_high[i] = max(max_high[i-1], high[i])
            min_low[i] = min(min_low[i-1], low[i])
    
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    # Align all 1d indicators to 12h timeframe
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        chop_val = chop[i]
        
        # Get current 1d volume (assuming ~2 bars per day for 12h timeframe)
        vol_idx = min(i // 2, len(df_1d)-1) if i >= 2 else 0
        volume = df_1d['volume'].iloc[vol_idx] if hasattr(df_1d['volume'], 'iloc') else df_1d['volume'][vol_idx]
        
        # Volume confirmation: current 1d volume > 2.0x 20-day average
        volume_confirm = volume > 2.0 * vol_ma if not (np.isnan(volume) or np.isnan(vol_ma)) else False
        
        # Chop regime: trending when chop < 38.2
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Long: Jaws < Teeth < Lips (bullish alignment) and price > lips
            if jaw_val < teeth_val < lips_val and price > lips_val and volume_confirm and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) and price < lips
            elif jaw_val > teeth_val > lips_val and price < lips_val and volume_confirm and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if alignment breaks or price crosses below lips
                if not (jaw_val < teeth_val < lips_val) or price < lips_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if alignment breaks or price crosses above lips
                if not (jaw_val > teeth_val > lips_val) or price > lips_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0