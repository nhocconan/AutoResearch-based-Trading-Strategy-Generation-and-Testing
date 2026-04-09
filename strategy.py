#!/usr/bin/env python3
# 1d_keltner_1w_trend_volume_v1
# Hypothesis: 1d strategy using 1w Keltner Channel (EMA20 ± 2*ATR10) for trend and volatility bands,
# with volume confirmation (>1.5x 20-period average) for entry timing.
# In bull markets: price touches upper band + volume spike → long
# In bear markets: price touches lower band + volume spike → short
# The Keltner Channel adapts to volatility, providing dynamic support/resistance.
# Discrete sizing (±0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for Keltner Channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Keltner Channel parameters
    ema_period = 20
    atr_period = 10
    multiplier = 2.0
    
    # EMA of close
    ema_close = pd.Series(close_1w).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # True Range and ATR
    tr1 = pd.Series(high_1w).diff().abs()
    tr2 = (pd.Series(high_1w) - pd.Series(close_1w).shift()).abs()
    tr3 = (pd.Series(low_1w) - pd.Series(close_1w).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Keltner Channel bands
    upper_band = ema_close + multiplier * atr
    lower_band = ema_close - multiplier * atr
    
    # Align Keltner components to 1d timeframe
    ema_close_aligned = align_htf_to_ltf(prices, df_1w, ema_close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_close_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below EMA (trend reversal)
            if close[i] < ema_close_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA (trend reversal)
            if close[i] > ema_close_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price touches or crosses above upper band
                if close[i] >= upper_band_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches or crosses below lower band
                elif close[i] <= lower_band_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals