#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Squeeze breakout with volume confirmation and weekly trend filter.
# Long when price breaks above upper Bollinger Band during low volatility (squeeze) + volume spike + price > weekly EMA50
# Short when price breaks below lower Bollinger Band during squeeze + volume spike + price < weekly EMA50
# Exit when price re-enters the Bollinger Bands or volatility expands (BB width > 1.5x average)
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Target: 15-30 trades/year to minimize fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Bollinger Bands (20, 2) on 12h closes
    close = prices['close'].values
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_width = upper - lower
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Squeeze: current width < 0.5 * 20-period average width
    squeeze = bb_width < 0.5 * bb_width_ma
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema50 = ema50_1w_aligned[i]
        sq = squeeze[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper BB during squeeze + volume spike + price > weekly EMA50
            if price > upper[i] and sq and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB during squeeze + volume spike + price < weekly EMA50
            elif price < lower[i] and sq and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price re-enters Bollinger Bands or volatility expands
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price re-enters below upper BB or volatility expands (BB width > 1.5x average)
                if price < upper[i] or bb_width[i] > 1.5 * bb_width_ma[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price re-enters above lower BB or volatility expands
                if price > lower[i] or bb_width[i] > 1.5 * bb_width_ma[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_BollingerSqueeze_Breakout_Volume_WeeklyEMA50"
timeframe = "12h"
leverage = 1.0