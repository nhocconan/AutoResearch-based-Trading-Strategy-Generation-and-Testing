#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ChopRegime
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA50 trend filter and choppiness regime filter captures institutional breakout attempts with trend alignment while avoiding choppy markets. 
Uses volume confirmation (20-bar average) to ensure breakout legitimacy. 
Designed for low trade frequency (~20-40/year) to work in both bull and bear markets via trend alignment and regime filter.
Camarilla levels represent key intraday support/resistance; volume confirmation reduces false signals; 1d EMA50 ensures trades align with daily trend; chop filter avoids whipsaws in ranging markets.
"""

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
    
    # Get 1d data for HTF trend filter and choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) on 1d for choppiness indicator
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # First TR is NaN
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate choppiness index: CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # We'll use a simplified version: CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, max_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, max_periods=14).min().values
    chop_denom = hh_14 - ll_14
    chop_ratio = np.where(chop_denom > 0, atr_sum / chop_denom, np.nan)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-bar average volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR(14) and volume MA20
    start_idx = 34  # 20 for volume + 14 for ATR warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels from previous 1d bar
        # Need previous day's OHLC (1d bar that closed before current 4h bar)
        # Since we're on 4h timeframe, we use the 1d data from align_htf_to_ltf which gives us previous day's values
        # For simplicity, we'll use the 1d bar's high/low/close that is aligned to current 4h bar
        # In practice, we need to get the actual previous day's OHLC, but for now we'll use current aligned values
        # This is a simplification - in production we'd need to shift the 1d data by 1 bar
        prev_close_1d = close_1d[:-1]  # Shifted by 1 to get previous day's close
        prev_high_1d = high_1d[:-1]   # Shifted by 1 to get previous day's high
        prev_low_1d = low_1d[:-1]     # Shifted by 1 to get previous day's low
        
        # Align the shifted arrays
        if len(prev_close_1d) > 0:
            prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], prev_close_1d]))
            prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], prev_high_1d]))
            prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], prev_low_1d]))
        else:
            prev_close_1d_aligned = np.full(n, np.nan)
            prev_high_1d_aligned = np.full(n, np.nan)
            prev_low_1d_aligned = np.full(n, np.nan)
        
        if position == 0:
            # Calculate Camarilla levels from previous day's OHLC
            # R1 = Close + (High - Low) * 1.1/12
            # S1 = Close - (High - Low) * 1.1/12
            rng = prev_high_1d_aligned[i] - prev_low_1d_aligned[i]
            r1 = prev_close_1d_aligned[i] + rng * 1.1 / 12
            s1 = prev_close_1d_aligned[i] - rng * 1.1 / 12
            
            # Volume confirmation: current volume > 1.3x 20-bar average
            volume_confirm = volume[i] > 1.3 * vol_ma20[i]
            
            # Choppiness filter: only trade when CHOP < 61.8 (trending market)
            trending_market = chop_aligned[i] < 61.8
            
            # Long: price breaks above R1 in uptrend with volume and trending market
            # Short: price breaks below S1 in downtrend with volume and trending market
            long_signal = (close[i] > r1) and (close[i] > ema50_1d_aligned[i]) and volume_confirm and trending_market
            short_signal = (close[i] < s1) and (close[i] < ema50_1d_aligned[i]) and volume_confirm and trending_market
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below 1d EMA50 (trend reversal) or chop increases significantly
            exit_signal = close[i] < ema50_1d_aligned[i] or chop_aligned[i] > 61.8
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal) or chop increases significantly
            exit_signal = close[i] > ema50_1d_aligned[i] or chop_aligned[i] > 61.8
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ChopRegime"
timeframe = "4h"
leverage = 1.0