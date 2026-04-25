#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dChopRegime_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d chop regime filter (Bollinger Band Width percentile < 30 = ranging) and volume confirmation (1.5x 20-bar avg). In ranging markets, mean reversion at extreme Camarilla levels works well. Volume confirms breakout validity. Designed for 4h timeframe targeting 20-40 trades/year. Works in bull/bear by fading extremes in ranging regimes.
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
    
    # Get 1d data for HTF regime filter (Bollinger Band Width)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Band Width (20, 2) on 1d data
    bb_period = 20
    bb_std = 2.0
    ma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = ma_1d + (bb_std * std_1d)
    lower_bb = ma_1d - (bb_std * std_1d)
    bb_width = (upper_bb - lower_bb) / ma_1d
    
    # Calculate Bollinger Band Width percentile (lookback 50 days) for regime
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Align BB Width percentile to 4h timeframe (1-day lagged for completed bar)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile, additional_delay_bars=1)
    
    # Regime: ranging when BB Width percentile < 30 (low volatility)
    ranging_regime = bb_width_percentile_aligned < 30
    
    # Calculate Camarilla levels on 1d data (based on previous day's OHLC)
    # Camarilla: R1 = C + ((H-L) * 1.1/12), S1 = C - ((H-L) * 1.1/12)
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for BB percentile and Camarilla
    start_idx = max(50, 30)  # 50 for BB percentile warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for mean reversion signals at R1/S1 in ranging regime with volume confirmation
            long_signal = (close[i] < camarilla_s1_aligned[i]) and ranging_regime[i] and volume_spike[i]
            short_signal = (close[i] > camarilla_r1_aligned[i]) and ranging_regime[i] and volume_spike[i]
            
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
            # Exit when price moves back above Camarilla C (mean reversion target)
            camarilla_c_1d = close_1d  # Camarilla C is close
            camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c_1d, additional_delay_bars=1)
            exit_signal = close[i] > camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back below Camarilla C (mean reversion target)
            camarilla_c_1d = close_1d  # Camarilla C is close
            camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c_1d, additional_delay_bars=1)
            exit_signal = close[i] < camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dChopRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0