#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_v1
Hypothesis: On 1h timeframe, trade Camarilla R1/S1 breakouts in direction of 4h EMA50 trend, with 1d Bollinger Band Width regime filter (BW < 30th percentile = low volatility/trending) to avoid whipsaws. Uses discrete sizing (0.20) and session filter (08-20 UTC) to limit trades to ~25/year. 4h EMA50 provides smooth trend alignment, reducing false breakouts. Volume confirmation (>1.5x 20-bar avg) ensures breakout momentum. Designed for BTC/ETH robustness: in bull markets, trend filter captures momentum; in bear/ranging markets, regime filter avoids chop and enables mean-reversion at extremes.
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
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for HTF regime filter (Camarilla levels and BB width)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior day)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate Bollinger Band Width for regime filter on 1d
    bb_middle_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper_1d = bb_middle_1d + 2.0 * bb_std_1d
    bb_lower_1d = bb_middle_1d - 2.0 * bb_std_1d
    bb_width_1d = (bb_upper_1d - bb_lower_1d) / bb_middle_1d
    # Use 30th percentile of BB width over 50 bars as regime threshold (low volatility = trending)
    bb_width_percentile_30 = pd.Series(bb_width_1d).rolling(window=50, min_periods=50).quantile(0.30).values
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    bb_width_percentile_30_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile_30)
    chop_filter = bb_width_1d_aligned < bb_width_percentile_30_aligned  # True when in low volatility (trending) regime
    
    # Calculate 20-bar average volume for confirmation on 1h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50, volume MA20, and BB calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma20[i]) or
            np.isnan(chop_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-bar average (moderate filter)
            volume_confirm = volume[i] > 1.5 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R1 in uptrend with volume spike and trending regime
            # Short: price breaks below Camarilla S1 in downtrend with volume spike and trending regime
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_4h_aligned[i]) and volume_confirm and chop_filter[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_4h_aligned[i]) and volume_confirm and chop_filter[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price moves back below 4h EMA50 (trend reversal)
            exit_signal = close[i] < ema50_4h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above 4h EMA50 (trend reversal)
            exit_signal = close[i] > ema50_4h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0