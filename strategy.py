#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily RSI for mean reversion signal ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily Moving Average for trend filter ===
    ma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # === Weekly Bollinger Bands for volatility regime ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = ma_20_1w + (2 * std_20_1w)
    lower_bb_1w = ma_20_1w - (2 * std_20_1w)
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / ma_20_1w
    bb_width_1w_avg = pd.Series(bb_width_1w).rolling(window=50, min_periods=50).mean().values
    bb_width_1w_avg_aligned = align_htf_to_ltf(prices, df_1w, bb_width_1w_avg)
    
    # === Daily Volume Filter ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ma_50[i]) or 
            np.isnan(bb_width_1w_avg_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ma50 = ma_50[i]
        bb_width_avg = bb_width_1w_avg_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # === EXIT LOGIC: Exit when RSI reverts or volatility regime changes ===
        if position == 1:  # Long position
            # Exit when RSI returns to neutral range
            if rsi_val >= 40:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when RSI returns to neutral range
            if rsi_val <= 60:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Oversold RSI + price above MA50 + low volatility regime + volume confirmation
            if (rsi_val < 30 and price > ma50 and 
                bb_width_avg < 0.03 and vol > 1.2 * vol_ma):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Overbought RSI + price below MA50 + low volatility regime + volume confirmation
            elif (rsi_val > 70 and price < ma50 and 
                  bb_width_avg < 0.03 and vol > 1.2 * vol_ma):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_RSI_MA50_BBWidth_VolumeFilter"
timeframe = "1d"
leverage = 1.0