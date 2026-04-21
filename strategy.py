#!/usr/bin/env python3
"""
1h_VolumeSpike_HTFTrend_Regime
Hypothesis: On 1h timeframe, enter only when 1h volume spikes (>2x 20-period average) AND 4h EMA50 trend aligns AND choppiness regime filter (CHOP < 50 for trending, > 50 for ranging) confirms market state. In trending regime (CHOP < 50): breakout in direction of 4h trend. In ranging regime (CHOP >= 50): mean reversion at 1h Bollinger Bands (20,2). Uses 0.20 position size to limit drawdown. Targets 15-35 trades/year by requiring volume spike + HTF alignment + regime filter. Works in bull/bear via HTF trend filter and regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for EMA50 trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for trend filter ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h Indicators (primary timeframe) ===
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * vol_ma
    
    # Bollinger Bands (20,2) for mean reversion in ranging regime
    sma_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Choppiness Index (14-period) for regime filter
    tr1 = pd.Series(high_1h - low_1h)
    tr2 = pd.Series(np.abs(high_1h - np.roll(close_1h, 1)))
    tr3 = pd.Series(np.abs(low_1h - np.roll(close_1h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    cp = pd.Series(high_1h - low_1h).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(cp / tr_sum) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_threshold[i]) 
            or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) 
            or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1h[i]
        
        if position == 0:
            # Regime filter: CHOP < 50 = trending, CHOP >= 50 = ranging
            in_trending = chop[i] < 50
            in_ranging = chop[i] >= 50
            
            # Volume spike condition
            volume_spike = volume_1h[i] > volume_threshold[i]
            
            if in_trending and volume_spike:
                # In trending regime: breakout in direction of 4h EMA50
                if price > ema_50_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                elif price < ema_50_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
            
            elif in_ranging and volume_spike:
                # In ranging regime: mean reversion at Bollinger Bands
                if price < lower_bb[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                elif price > upper_bb[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Exit conditions: reverse signal or stoploss
            if price < ema_50_4h_aligned[i] and chop[i] < 50:  # Trend reversal
                signals[i] = 0.0
                position = 0
            elif price > sma_20[i] and chop[i] >= 50:  # Mean reversion to mid-band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit conditions: reverse signal or stoploss
            if price > ema_50_4h_aligned[i] and chop[i] < 50:  # Trend reversal
                signals[i] = 0.0
                position = 0
            elif price < sma_20[i] and chop[i] >= 50:  # Mean reversion to mid-band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_HTFTrend_Regime"
timeframe = "1h"
leverage = 1.0