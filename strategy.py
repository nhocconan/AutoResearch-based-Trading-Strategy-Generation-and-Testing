#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter, volume spike confirmation, and choppiness regime filter.
Only trades when market is trending (CHOP < 38.2) to avoid whipsaws in ranging markets.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels on 1d data (based on previous bar's OHLC)
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    camarilla_c_1d = close_1d  # Camarilla C is the close
    
    # Align HTF indicators to 4h timeframe (completed 1d bar lag)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c_1d, additional_delay_bars=1)
    
    # Calculate choppiness index on 1d data for regime filter
    def choppiness_index(high, low, close, window=14):
        atr = []
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.concatenate([[np.nan], pd.Series(tr).rolling(window, min_periods=1).mean().values])
        high_low = np.abs(np.max(high) - np.min(low)) if len(high) > 0 else 0
        sum_atr = np.nansum(atr[-window:]) if len(atr) >= window else np.nan
        if np.isnan(sum_atr) or sum_atr == 0:
            return 100.0
        chop = 100 * np.log10(sum_atr / np.log10(window)) / np.log10(window)
        return chop
    
    chop_values = []
    for i in range(len(close_1d)):
        start = max(0, i - 13)
        end = i + 1
        if end - start < 14:
            chop_values.append(np.nan)
        else:
            h_slice = high_1d[start:end]
            l_slice = low_1d[start:end]
            c_slice = close_1d[start:end]
            chop_values.append(choppiness_index(h_slice, l_slice, c_slice))
    chop_1d = np.array(chop_values)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 and chop
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_c_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0 and is_trending:
            # Look for breakout signals in direction of 1d trend with volume confirmation
            # Long: price breaks above R1 in uptrend (close > EMA34)
            # Short: price breaks below S1 in downtrend (close < EMA34)
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema34_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema34_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below Camarilla C (mean reversion to midpoint) OR regime changes to ranging
            exit_signal = close[i] < camarilla_c_aligned[i] or chop_aligned[i] >= 38.2
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla C (mean reversion to midpoint) OR regime changes to ranging
            exit_signal = close[i] > camarilla_c_aligned[i] or chop_aligned[i] >= 38.2
            if exit_signal:
                signals[i] = 0.0
                position = 0
        else:
            # In ranging market or no signal, stay flat or hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0