#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and 12h trend filter
# Long when price touches Camarilla S3 level + volume > 2x average + 12h trend up
# Short when price touches Camarilla R3 level + volume > 2x average + 12h trend down
# Exit when price reaches Camarilla C level (close) or trend reverses
# Designed for 30-60 trades/year on 4h timeframe with mean reversion in range markets

name = "4h_12h_camarilla_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(25) for trend filter
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla formulas: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.125*(high-low)
    # H2 = close + 0.75*(high-low)
    # H1 = close + 0.5*(high-low)
    # L1 = close - 0.5*(high-low)
    # L2 = close - 0.75*(high-low)
    # L3 = close - 1.125*(high-low)
    # L4 = close - 1.5*(high-low)
    # C = close (pivot)
    
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_c = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous 12h bar's OHLC (need to map 12h to 4h)
        # Since we're using 12h data aligned to 4h, we can use the aligned values
        pass  # Will calculate properly below
    
    # Instead, calculate Camarilla from 12h OHLC and align to 4h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_h3_12h = close_12h + 1.125 * (high_12h - low_12h)
    camarilla_l3_12h = close_12h - 1.125 * (high_12h - low_12h)
    camarilla_c_12h = close_12h  # pivot point
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3_12h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3_12h)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_12h, camarilla_c_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(25, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_c_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_25_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trend filter: price relative to 12h EMA25
        is_uptrend = close[i] > ema_25_12h_aligned[i]
        is_downtrend = close[i] < ema_25_12h_aligned[i]
        
        # Entry conditions: price touches Camarilla S3/L3 with volume and trend alignment
        # For long: price touches or goes below S3 (L3) in uptrend (mean reversion long)
        # For short: price touches or goes above R3 (H3) in downtrend (mean reversion short)
        long_entry = (close[i] <= camarilla_l3_aligned[i]) and volume_filter and is_uptrend
        short_entry = (close[i] >= camarilla_h3_aligned[i]) and volume_filter and is_downtrend
        
        # Exit conditions: price returns to Camarilla C level (pivot) or trend reverses
        long_exit = (close[i] >= camarilla_c_aligned[i]) or (not is_uptrend)
        short_exit = (close[i] <= camarilla_c_aligned[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals