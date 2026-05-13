#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_With_Volume_Spike
Hypothesis: Keltner Channel (ATR-based volatility bands) captures volatility expansion during breakouts. A close above/below the upper/lower band with volume > 2x average and aligned weekly trend (close > EMA50) signals trend continuation. Uses 25% position size to limit risk and trade frequency (~10-25/year) to minimize fee drag in daily bars.
"""

name = "1d_Keltner_Channel_Breakout_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ATR (14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate EMA (20) for Keltner Channel middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Weekly trend filter: EMA(50) on weekly close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after EMA20 warmup
        if position == 0:
            # LONG: Close above upper Keltner band, volume spike, price above weekly EMA50 (uptrend)
            if (close[i] > kc_upper[i] and 
                volume_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below lower Keltner band, volume spike, price below weekly EMA50 (downtrend)
            elif (close[i] < kc_lower[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below middle line (EMA20) OR volume drops
            if (close[i] < ema20[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above middle line (EMA20) OR volume drops
            if (close[i] > ema20[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals