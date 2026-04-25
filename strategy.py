#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime_v1
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter, volume spike (>2x 20-bar avg), and chop regime filter (CHOP > 61.8 for range, < 38.2 for trend). Uses discrete sizing (0.30) to target ~30 trades/year. Designed for BTC/ETH robustness: in trending regimes (CHOP < 38.2) follow breakout direction; in ranging regimes (CHOP > 61.8) fade breaks at R1/S1 with mean reversion to pivot (PP). Volume spike confirms momentum. Avoids overtrading via tight entry conditions and regime filter.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC of prior bar)
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_r1 = camarilla_pp + 1.1 * (high_1d - low_1d) / 2
    camarilla_s1 = camarilla_pp - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 20-bar average volume for confirmation on 4h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 4h for regime filter
    def choppiness_index(high, low, close, window=14):
        atr = np.zeros(len(high))
        atr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34, volume MA20, chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Regime logic
            if chop[i] < 38.2:  # Trending regime
                # Follow breakout direction with trend filter
                long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema34_1d_aligned[i]) and volume_confirm
                short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema34_1d_aligned[i]) and volume_confirm
            elif chop[i] > 61.8:  # Ranging regime
                # Fade breaks at R1/S1 with mean reversion to pivot
                long_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] > camarilla_pp_aligned[i]) and volume_confirm
                short_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] < camarilla_pp_aligned[i]) and volume_confirm
            else:  # Neutral regime (38.2 <= CHOP <= 61.8) - no trading
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit when price moves back below pivot (mean reversion) or above R1 (take profit)
            exit_signal = (close[i] < camarilla_pp_aligned[i]) or (close[i] > camarilla_r1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit when price moves back above pivot (mean reversion) or below S1 (take profit)
            exit_signal = (close[i] > camarilla_pp_aligned[i]) or (close[i] < camarilla_s1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0