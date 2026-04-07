#!/usr/bin/env python3
"""
6h_trix_1d_volume_regime_v1
Hypothesis: On 6-hour timeframe, use TRIX momentum oscillator for entry signals filtered by daily volume spike and 6-hour chop regime (range vs trend). TRIX filters out minor cycles and is effective in both trending and ranging markets when combined with volume confirmation. Volume spikes indicate institutional interest and increase probability of sustained moves. Chop regime filter avoids whipsaw in sideways markets by switching between mean reversion (in chop) and trend following (in trend). Target: 25-35 trades/year to minimize fee drag while capturing high-probability moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_1d_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX on 6h (1-period ROC of triple EMA)
    # TRIX = 100 * (EMA3 of ROC)
    close_s = pd.Series(close)
    roc = close_s.pct_change(1)  # 1-period rate of change
    ema1 = roc.ewm(span=15, min_periods=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, min_periods=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, min_periods=15, adjust=False).mean()
    trix = 100 * ema3
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day average volume on daily
    d_volume = df_1d['volume'].values
    d_vol_avg = pd.Series(d_volume).rolling(window=20, min_periods=20).mean().values
    d_vol_avg_aligned = align_htf_to_ltf(prices, df_1d, d_vol_avg)
    
    # Calculate chop regime on 6h (using ATR and price range)
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 14-period high-low range
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    # Chop = 100 * log10(sum(ATR14) / (maxHigh-minLow)) / log10(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where(range_hl > 0, 100 * np.log10(atr_sum / range_hl) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(trix.iloc[i]) or np.isnan(d_vol_avg_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5x daily average volume
        vol_spike = volume[i] > 1.5 * d_vol_avg_aligned[i]
        
        # Regime filters
        in_chop = chop[i] > 61.8  # Choppy market
        in_trend = chop[i] < 38.2  # Trending market
        
        # TRIX signals
        trix_cross_up = trix.iloc[i] > 0 and trix.iloc[i-1] <= 0
        trix_cross_down = trix.iloc[i] < 0 and trix.iloc[i-1] >= 0
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when TRIX crosses below zero
            if trix_cross_down:
                exit_long = True
            # Exit when volatility spikes against position
            elif trix.iloc[i] < -0.5:  # Strong bearish momentum
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when TRIX crosses above zero
            if trix_cross_up:
                exit_short = True
            # Exit when volatility spikes against position
            elif trix.iloc[i] > 0.5:  # Strong bullish momentum
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry logic adapts to regime
            if in_trend:
                # Trend following: enter in direction of TRIX momentum
                long_entry = trix_cross_up and vol_spike
                short_entry = trix_cross_down and vol_spike
            elif in_chop:
                # Mean reversion: fade extreme TRIX readings
                long_entry = (trix.iloc[i] < -0.3 and trix.iloc[i] > trix.iloc[i-1]) and vol_spike
                short_entry = (trix.iloc[i] > 0.3 and trix.iloc[i] < trix.iloc[i-1]) and vol_spike
            else:
                # Neutral regime: standard TRIX crossover with volume
                long_entry = trix_cross_up and vol_spike
                short_entry = trix_cross_down and vol_spike
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals