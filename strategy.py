#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1d EMA(50) > EMA(200) AND 12h volume > 2.0x 20-bar avg
# - Short when price breaks below Camarilla L3 level AND 1d EMA(50) < EMA(200) AND 12h volume > 2.0x 20-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla levels provide institutional support/resistance; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation (2.0x avg) filters low-liquidity false breakouts
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakout strategy captures momentum in trends, pivot exit works in ranges

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1d EMA trend to 12h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Pre-compute Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h3 = pivot + (range_1d * 1.1 / 4)  # H3 = pivot + 1.1*(H-L)/4
    l3 = pivot - (range_1d * 1.1 / 4)  # L3 = pivot - 1.1*(H-L)/4
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Pre-compute 12h volume confirmation: > 2.0x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 1d bullish trend AND volume spike
            if (prices['high'].iloc[i] > h3_aligned[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 1d bearish trend AND volume spike
            elif (prices['low'].iloc[i] < l3_aligned[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla pivot (mean reversion)
            # Exit when price returns to pivot point
            if position == 1:
                exit_signal = prices['low'].iloc[i] <= pivot_aligned[i]
            else:  # position == -1
                exit_signal = prices['high'].iloc[i] >= pivot_aligned[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals