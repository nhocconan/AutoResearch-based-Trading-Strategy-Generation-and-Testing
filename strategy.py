#!/usr/bin/env python3
"""
12h Camarilla Pivot H3/L3 Breakout with Daily EMA Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Breakouts above H3 or below L3 
with daily EMA trend alignment and volume confirmation capture strong moves. 12h timeframe targets 
12-37 trades/year, reducing fee drag. Works in bull/bear by trading with daily trend filter.
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (H3, L3) from previous daily bar
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 12h timeframe (no extra delay - pivot levels known at bar open)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h ATR(14) for volatility normalization (optional filter)
    tr1 = pd.Series(high).sub(pd.Series(low))
    tr2 = pd.Series(high).sub(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).sub(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average (strict)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(34, 20) + 1  # EMA34 + volume MA + 1 for pivot shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Daily trend filter: price above/below EMA34
        uptrend = ema_34_aligned[i] is not None and curr_close > ema_34_aligned[i]
        downtrend = ema_34_aligned[i] is not None and curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla H3 AND uptrend AND volume spike
            long_entry = (curr_high > camarilla_h3_aligned[i]) and uptrend and vol_spike
            # Short: price breaks below Camarilla L3 AND downtrend AND volume spike
            short_entry = (curr_low < camarilla_l3_aligned[i]) and downtrend and vol_spike
            
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
            # Exit: price falls below Camarilla L3 OR loss of trend (price < EMA34)
            if (curr_low < camarilla_l3_aligned[i]) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 OR loss of trend (price > EMA34)
            if (curr_high > camarilla_h3_aligned[i]) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0