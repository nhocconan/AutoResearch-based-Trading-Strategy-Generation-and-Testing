#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance. Breakouts above R3 or below S3, when aligned with 1d EMA34 trend and confirmed by volume spikes, capture institutional momentum. This strategy targets 20-50 trades/year by requiring confluence of three strong filters, works in bull (long R3 breakouts) and bear (short S3 breakouts) regimes, and uses discrete position sizing (0.30) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous day (requires high, low, close of prior 1d bar)
    # We'll compute Camarilla for each 4h bar using the prior completed 1d bar
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # For each 4h bar, use the prior completed 1d bar's OHLC
    for i in range(n):
        # Find index of prior completed 1d bar in df_1d
        # We need the 1d bar that closed before or at the current 4h bar's open_time
        # Since align_htf_to_ltf gives us the completed 1d bar values, we can use a simpler approach:
        # The Camarilla levels are constant within a 1d bar, so we shift the 1d values to align with 4h
        pass  # We'll compute Camarilla differently below
    
    # Instead, compute Camarilla levels per 1d bar, then align to 4h
    # Camarilla formulas:
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    # Where C, H, L are close, high, low of prior day
    
    # We need prior day's OHLC, so we shift 1d data by 1
    if len(df_1d) >= 2:
        prior_close = df_1d['close'].shift(1).values
        prior_high = df_1d['high'].shift(1).values
        prior_low = df_1d['low'].shift(1).values
        
        # Calculate Camarilla R3 and S3 for each 1d bar
        rng = prior_high - prior_low
        camarilla_r3_1d = prior_close + rng * 1.1 / 4
        camarilla_s3_1d = prior_close - rng * 1.1 / 4
        
        # Align to 4h timeframe (values constant throughout the 4h bars within each 1d bar)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    else:
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 20)  # EMA34 lookback, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Camarilla breakout + trend + volume
            # Long: price breaks above Camarilla R3 AND bullish bias AND volume spike
            long_entry = (curr_high > camarilla_r3_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Camarilla S3 AND bearish bias AND volume spike
            short_entry = (curr_low < camarilla_s3_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla S3 (mean reversion) OR loss of bullish bias
            if (curr_low < camarilla_s3_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla R3 (mean reversion) OR loss of bearish bias
            if (curr_high > camarilla_r3_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0