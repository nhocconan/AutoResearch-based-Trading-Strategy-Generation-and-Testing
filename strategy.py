#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + Weekly RSI mean reversion.
# Choppiness Index (CHOP) > 61.8 indicates ranging market (mean revert),
# CHOP < 38.2 indicates trending market (trend follow). Weekly RSI extremes
# (>70 or <30) provide entry signals in the direction of mean reversion during
# ranging markets. This avoids trending whipsaws and focuses on mean reversion
# in chop, which works in both bull and bear markets. Low trade frequency
# expected due to dual regime + RSI extreme filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for RSI (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period RSI on weekly close
    delta = pd.Series(close_1w).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.fillna(50).values  # fill NaN with neutral 50
    
    # Load daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Choppiness Index on daily data
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    # Sum of True Range over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(TR_sum / (max_high - min_low)) / log10(14)
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)  # small epsilon
    chop = 100 * np.log10(tr_sum_14 / range_14) / np.log10(14)
    chop = chop.fillna(50).values  # fill NaN with neutral 50
    
    # Align weekly RSI and daily Chop to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        rsi_val = rsi_1w_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filters
        ranging_market = chop_val > 61.8  # CHOP > 61.8 = ranging (mean revert)
        trending_market = chop_val < 38.2  # CHOP < 38.2 = trending (avoid)
        
        if position == 0:
            # Only trade in ranging markets
            if ranging_market:
                # Long when weekly RSI is oversold (<30)
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                # Short when weekly RSI is overbought (>70)
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral (50) or market starts trending
                if rsi_val >= 50 or not ranging_market:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral (50) or market starts trending
                if rsi_val <= 50 or not ranging_market:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Choppiness_WeeklyRSI_MeanRev"
timeframe = "1d"
leverage = 1.0