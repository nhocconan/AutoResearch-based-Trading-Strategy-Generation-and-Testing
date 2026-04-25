#!/usr/bin/env python3
"""
4h Camarilla Pivot H3L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Breakouts above H3 or below L3 
with 1d EMA34 trend alignment, volume confirmation, and choppy market filter (CHOP > 61.8) capture 
strong momentum moves in both bull and bear markets. Chop filter avoids whipsaws in ranging markets.
Target: 20-40 trades/year on 4h to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    # Range = high - low
    # H3 = close + (high - low) * 1.1 / 4
    # L3 = close - (high - low) * 1.1 / 4
    typical_price = (high + low + close) / 3.0
    daily_range = high - low
    H3 = typical_price + daily_range * 1.1 / 4.0
    L3 = typical_price - daily_range * 1.1 / 4.0
    
    # Align H3/L3 to 4h timeframe (use previous day's levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index filter: CHOP > 61.8 = ranging market (avoid trading)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # Simplified: use rolling ATR and max/min range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate rolling max/min for CHOP
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) > 0, 
                    100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14), 
                    50)
    # Handle edge cases
    chop = np.where(np.isnan(chop), 50, chop)
    chop_filter = chop > 61.8  # Only trade in choppy markets (mean reversion prone)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 14, 34)  # volume MA, ATR, EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_choppy = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend + volume + chop filter
            # Long: break above H3 AND bullish bias AND volume spike AND choppy market
            long_entry = (curr_high > H3_aligned[i]) and bullish_bias and vol_spike and is_choppy
            # Short: break below L3 AND bearish bias AND volume spike AND choppy market
            short_entry = (curr_low < L3_aligned[i]) and bearish_bias and vol_spike and is_choppy
            
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
            # Exit: price crosses below L3 (mean reversion) OR loss of bullish bias
            if (curr_close < L3_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 (mean reversion) OR loss of bearish bias
            if (curr_close > H3_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0