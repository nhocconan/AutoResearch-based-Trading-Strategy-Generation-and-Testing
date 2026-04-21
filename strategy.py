#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + choppiness regime (proven pattern)
# Long when price crosses above R1 in low chop regime, short when crosses below S1 in low chop regime
# Volume spike (>2x 20-period average) confirms breakout strength
# Chop regime filter: Choppiness Index < 38.2 indicates trending market (avoid ranging)
# Target: 20-40 trades/year by requiring confluence of pivot break, volume, and trend
# Works in bull/bear: Choppiness filter avoids whipsaws in ranging markets, pivot levels provide clear structure

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Calculate ATR(14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilder_smooth(tr, 14)
    
    # Calculate Choppiness Index: 100 * log(sum(ATR14) / (max(high) - min(low))) / log(14)
    chop = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(13, len(close_1d)):
        if not np.isnan(atr_14[i]):
            sum_atr = np.nansum(atr_14[i-13:i+1])
            max_high = np.nanmax(high_1d[i-13:i+1])
            min_low = np.nanmin(low_1d[i-13:i+1])
            if max_high > min_low and sum_atr > 0:
                chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d OHLC for Camarilla pivot levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    hl_range = h_1d - l_1d
    r1 = c_1d + (hl_range * 1.1 / 12)
    s1 = c_1d - (hl_range * 1.1 / 12)
    
    # Align pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(chop_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Chop filter: trending market (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            if volume_confirm and trending_regime:
                # Long: price crosses above R1
                if price > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price crosses below S1
                elif price < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below S1 (failed breakout) or chop increases (ranging)
                if price < s1_aligned[i] or chop_aligned[i] > 50.0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above R1 (failed breakdown) or chop increases (ranging)
                if price > r1_aligned[i] or chop_aligned[i] > 50.0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dChop14_Volume"
timeframe = "4h"
leverage = 1.0