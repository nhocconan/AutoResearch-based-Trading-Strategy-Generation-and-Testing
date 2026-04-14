#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h EMA crossover with volume confirmation and 1d trend filter
# Uses EMA9/21 crossover on 4h, confirmed by volume > 1.3x 20-period average
# Trend filter: 1d EMA50 (bullish if price > EMA50, bearish if price < EMA50)
# Position sizing: 0.30 for strong signals, 0.15 for weaker signals
# Target: 25-35 trades/year by requiring trend alignment and volume confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data once
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 21 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA9 and EMA21 for trend
    close_12h = df_12h['close'].values
    ema_9_12h = pd.Series(close_12h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h EMA9 and EMA21 for entry timing
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align higher timeframe indicators
    ema_9_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_9_12h)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any data is NaN
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or 
            np.isnan(ema_9_12h_aligned[i]) or np.isnan(ema_21_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # EMA crossover signals on 4h
        ema9_above_ema21 = ema_9[i] > ema_21[i]
        ema9_below_ema21 = ema_9[i] < ema_21[i]
        
        # 12h trend alignment
        trend_12h_bullish = ema_9_12h_aligned[i] > ema_21_12h_aligned[i]
        trend_12h_bearish = ema_9_12h_aligned[i] < ema_21_12h_aligned[i]
        
        # 1d trend filter
        price_above_1d_ema50 = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema50 = close[i] < ema_50_1d_aligned[i]
        
        # Long conditions: bullish alignment across timeframes
        long_setup = (ema9_above_ema21 and 
                     trend_12h_bullish and 
                     price_above_1d_ema50 and
                     volume_confirm)
        
        # Short conditions: bearish alignment across timeframes
        short_setup = (ema9_below_ema21 and 
                      trend_12h_bearish and 
                      price_below_1d_ema50 and
                      volume_confirm)
        
        # Enter long with full size
        if long_setup:
            signals[i] = 0.30
        # Enter short with full size
        elif short_setup:
            signals[i] = -0.30
        # Exit when 4h EMA crossover reverses
        elif ema9_above_ema21 and signals[i-1] < 0:  # was short, now bullish cross
            signals[i] = 0.0
        elif ema9_below_ema21 and signals[i-1] > 0:  # was long, now bearish cross
            signals[i] = 0.0
        # Gradual exit when trend weakens
        elif signals[i-1] > 0 and not (price_above_1d_ema50 and trend_12h_bullish):
            signals[i] = signals[i-1] * 0.5  # reduce position
        elif signals[i-1] < 0 and not (price_below_1d_ema50 and trend_12h_bearish):
            signals[i] = signals[i-1] * 0.5  # reduce position
        else:
            # Hold current position
            signals[i] = signals[i-1]
    
    return signals

name = "4h_12h_1d_EMA_Alignment_Volume"
timeframe = "4h"
leverage = 1.0