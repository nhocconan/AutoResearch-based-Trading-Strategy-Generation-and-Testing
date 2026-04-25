#!/usr/bin/env python3
"""
1d Camarilla R1S1 Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: On daily timeframe, Camarilla R1/S1 levels act as strong intraday support/resistance.
Price breaking these levels with weekly EMA34 trend alignment and volume confirmation captures
institutional breakouts. Works in both bull/bear markets via discrete sizing (0.25) and trend filter.
Primary timeframe: 1d. HTF: 1w for EMA34 trend filter.
Target trades: 30-100 over 4 years (7-25/year).
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
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Camarilla levels from previous 1d bar
    def calculate_camarilla(high, low, close):
        range_val = high - low
        return {
            'R1': close + range_val * 1.0833,
            'S1': close - range_val * 1.0833,
            'PP': (high + low + close) / 3
        }
    
    camarilla_history = []
    for i in range(len(prices)):
        h = high[i]
        l = low[i]
        c = close[i]
        camarilla_history.append(calculate_camarilla(h, l, c))
    
    camarilla_df = pd.DataFrame(camarilla_history)
    r1 = camarilla_df['R1'].values
    s1 = camarilla_df['S1'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and 1w EMA warmup
    start_idx = max(35, 20)  # EMA34 needs ~34, vol MA 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA34
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend + volume
            # Long: price breaks above R1 AND bullish bias AND volume spike
            long_entry = (curr_high > r1[i]) and bullish_bias and vol_spike
            # Short: price breaks below S1 AND bearish bias AND volume spike
            short_entry = (curr_low < s1[i]) and bearish_bias and vol_spike
            
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
            # Exit: price falls below S1 (mean reversion) OR loss of bullish bias
            if (curr_low < s1[i]) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above R1 (mean reversion) OR loss of bearish bias
            if (curr_high > r1[i]) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0