#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation
    # Williams %R identifies overbought/oversold conditions. In trending markets (filtered by 1d EMA50),
    # extreme readings followed by reversal provide high-probability mean reversion entries.
    # Volume confirmation ensures institutional participation. This works in both bull and bear markets
    # by only taking mean reversion trades in the direction of the higher timeframe trend.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R on 12h timeframe (period=14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after EMA and Williams %R warmup
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume + price above 1d EMA50 (uptrend)
            if williams_r[i] < -80 and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume + price below 1d EMA50 (downtrend)
            elif williams_r[i] > -20 and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral zone (-50) or trend reversal vs 1d EMA50
            if position == 1:
                if williams_r[i] > -50 or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r[i] < -50 or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0