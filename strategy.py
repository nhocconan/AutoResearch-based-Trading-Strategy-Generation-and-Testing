#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above H3 pivot level AND 1w EMA(21) > EMA(55) (bullish trend) AND 12h volume > 1.8x 30-bar avg
# - Short when price breaks below L3 pivot level AND 1w EMA(21) < EMA(55) (bearish trend) AND 12h volume > 1.8x 30-bar avg
# - Exit when price returns to the daily pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivots identify key intraday support/resistance levels
# - 1w EMA filter ensures alignment with weekly trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false breakouts
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakout trading in trends, pivot mean reversion in ranges

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(21) vs EMA(55)
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_55 = pd.Series(close_1w).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_bullish = ema_21 > ema_55
    ema_bearish = ema_21 < ema_55
    
    # Align 1w EMA trend to 12h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish)
    
    # Pre-compute daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculations
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H3 = Pivot + (Range * 1.1/2)
    # L3 = Pivot - (Range * 1.1/2)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    h3 = pivot + (rng * 1.1 / 2.0)
    l3 = pivot - (rng * 1.1 / 2.0)
    
    # Align daily Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Pre-compute 12h volume confirmation: > 1.8x 30-period average
    volume = prices['volume'].values
    volume_30_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (1.8 * volume_30_avg)
    
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
            # Long when price breaks above H3 AND 1w bullish trend AND volume spike
            if (prices['close'].iloc[i] > h3_aligned[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 1w bearish trend AND volume spike
            elif (prices['close'].iloc[i] < l3_aligned[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to daily pivot (mean reversion)
            # Exit when price returns to the daily pivot point
            exit_signal = np.abs(prices['close'].iloc[i] - pivot_aligned[i]) < (0.1 * (h3_aligned[i] - l3_aligned[i]))
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals