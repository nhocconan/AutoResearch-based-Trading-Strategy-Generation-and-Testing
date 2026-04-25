#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance.
Breakouts above H3 or below L3 with 1d EMA34 trend alignment and volume confirmation
capture momentum moves while avoiding choppy markets. The 12h timeframe reduces
fee drag, and the 1d EMA34 filter ensures we trade with the higher timeframe trend.
Designed to work in both bull (breakouts with trend) and bear (mean reversion at extremes) markets.
Target: 12-30 trades/year on 12h to stay within fee drag limits.
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
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 12h bar
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # We use the previous completed 12h bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan  # first bar has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after first bar to have previous bar data
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma[i])):
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
            # Look for breakout entries - require: Camarilla breakout + trend + volume
            # Long: break above H3 AND bullish bias AND volume spike
            long_entry = (curr_high > camarilla_h3[i]) and bullish_bias and vol_spike
            # Short: break below L3 AND bearish bias AND volume spike
            short_entry = (curr_low < camarilla_l3[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price re-enters Camarilla range (between H3 and L3) OR loss of bullish bias
            camarilla_h4 = camarilla_h3[i] + 1.1 * (camarilla_h3[i] - camarilla_l3[i]) / 2  # H4 for reference
            camarilla_l4 = camarilla_l3[i] - 1.1 * (camarilla_h3[i] - camarilla_l3[i]) / 2  # L4 for reference
            # Exit if price moves back below H3 (failed breakout) or above H4 (extreme)
            if (curr_close < camarilla_h3[i]) or (curr_close > camarilla_h4[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price re-enters Camarilla range OR loss of bearish bias
            camarilla_h4 = camarilla_h3[i] + 1.1 * (camarilla_h3[i] - camarilla_l3[i]) / 2
            camarilla_l4 = camarilla_l3[i] - 1.1 * (camarilla_h3[i] - camarilla_l3[i]) / 2
            # Exit if price moves back above L3 (failed breakdown) or below L4 (extreme)
            if (curr_close > camarilla_l3[i]) or (curr_close < camarilla_l4[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0