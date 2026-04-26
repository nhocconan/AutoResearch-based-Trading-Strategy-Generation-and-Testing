#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_ChopRegime
Hypothesis: Camarilla R3/S3 breakouts filtered by 1d EMA34 trend and choppiness regime (CHOP > 61.8 = range, < 38.2 = trend). In trending regimes (CHOP < 38.2), trade breakouts in trend direction. In ranging regimes (CHOP > 61.8), fade moves at R3/S3 levels. Volume confirmation (>1.5x 20-bar MA) reduces false signals. Designed for 4h timeframe to balance trade frequency and edge. Works in bull/bear markets by adapting to regime: trend-following in trends, mean-reversion in ranges. Target: 20-40 trades/year (80-160 total over 4 years).
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)  # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Choppiness Index: CHOP > 61.8 = range, CHOP < 38.2 = trend
    # CHOP = 100 * LOG10(SUM(ATR(1), 14) / (MAXHIGH(14) - MINLOW(14))) / LOG10(14)
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    atr = np.array(atr_list)
    
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(atr_sum / (max_high - min_low)) / np.log10(14))
    chop = np.where((max_high - min_low) == 0, 50, chop)  # avoid division by zero
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 34 for 1d EMA, 14 for chop)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # Determine regime: trending if CHOP < 38.2, ranging if CHOP > 61.8, neutral otherwise
        trending_regime = chop_val < 38.2
        ranging_regime = chop_val > 61.8
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        if position == 0:
            # Flat - look for entry
            if trending_regime:
                # Trend-following: breakout in trend direction
                long_entry = (close_val > camarilla_r3_val) and bullish_1d and vol_spike
                short_entry = (close_val < camarilla_s3_val) and bearish_1d and vol_spike
                if long_entry:
                    signals[i] = base_size
                    position = 1
                elif short_entry:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = 0.0
            elif ranging_regime:
                # Mean-reversion: fade at extremes
                long_entry = (close_val < camarilla_s3_val) and vol_spike  # buy at support
                short_entry = (close_val > camarilla_r3_val) and vol_spike  # sell at resistance
                if long_entry:
                    signals[i] = base_size
                    position = 1
                elif short_entry:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral regime - no trade
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            if trending_regime:
                # Exit on trend reversal or mean reversion to mid-point
                mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
                if close_val < mid_point or not bullish_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size
            elif ranging_regime:
                # Exit on reversion to mid-point or opposite extreme
                mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
                if close_val > mid_point:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size
            else:
                # Neutral - hold
                signals[i] = base_size
        elif position == -1:
            # Short - exit conditions
            if trending_regime:
                # Exit on trend reversal or mean reversion to mid-point
                mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
                if close_val > mid_point or not bearish_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
            elif ranging_regime:
                # Exit on reversion to mid-point or opposite extreme
                mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
                if close_val < mid_point:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
            else:
                # Neutral - hold
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_ChopRegime"
timeframe = "4h"
leverage = 1.0