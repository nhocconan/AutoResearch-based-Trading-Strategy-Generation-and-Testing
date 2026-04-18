#!/usr/bin/env python3
"""
4h Bollinger Band Squeeze Breakout with Volume Spike and 1d Trend Filter
Hypothesis: Bollinger Band squeeze (low volatility) followed by breakout with volume confirmation
and alignment with higher timeframe trend (1d EMA50) captures explosive moves in both bull and bear markets.
Low trade frequency (<30/year) minimizes fee drag while capturing high-momentum moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    
    # Bollinger Squeeze: BB width < 20-period percentile 10 (low volatility)
    bb_width_series = pd.Series(bb_width)
    bb_percentile = bb_width_series.rolling(window=20, min_periods=20).quantile(0.10).values
    squeeze = bb_width < bb_percentile
    
    # Volume spike: volume > 2.0 x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for BB, volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_squeeze = squeeze[i]
        vol_spike_bar = vol_spike[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # Look for breakout after squeeze with volume spike
            if bb_squeeze and vol_spike_bar:
                # Break above upper BB = long (only if above 1d EMA50)
                if price > upper_bb[i] and price > ema50:
                    signals[i] = 0.30
                    position = 1
                # Break below lower BB = short (only if below 1d EMA50)
                elif price < lower_bb[i] and price < ema50:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit: price returns to middle BB or volatility expands (no longer squeezed)
            if price < sma20[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price returns to middle BB or volatility expands
            if price > sma20[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Bollinger_Squeeze_Breakout_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0