#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with 1d volume confirmation and 1w trend filter
# Long when price closes above upper BB(20,2) AND volume > 1.5x 20-day avg volume AND 1w EMA50 > 1w EMA200
# Short when price closes below lower BB(20,2) AND volume > 1.5x 20-day avg volume AND 1w EMA50 < 1w EMA200
# Exit when price re-enters the Bollinger Bands (mean reversion in squeeze)
# Bollinger squeeze identifies low volatility breakouts, volume confirms breakout strength, weekly EMA trend filters for direction
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Bollinger Bands (20,2) ===
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # === 1d Volume Confirmation (20-day average volume) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1w Trend Filter: EMA50 vs EMA200 ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        vol_ma = vol_ma_20[i]
        ema_50 = ema_50_1w_aligned[i]
        ema_200 = ema_200_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average volume
        vol_confirm = volume[i] > vol_ma * 1.5
        
        # Trend filter: EMA50 > EMA200 for long, EMA50 < EMA200 for short
        trend_long = ema_50 > ema_200
        trend_short = ema_50 < ema_200
        
        # === EXIT LOGIC: price re-enters Bollinger Bands ===
        if position == 1:  # Long position
            if price <= bb_up:  # Price re-entered upper band
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if price >= bb_low:  # Price re-entered lower band
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price closes above upper BB AND volume confirmation AND weekly uptrend
            if price > bb_up and vol_confirm and trend_long:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price closes below lower BB AND volume confirmation AND weekly downtrend
            elif price < bb_low and vol_confirm and trend_short:
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

name = "1d_BB20_2_Squeeze_Volume1.5x_1wEMA50_200_Trend"
timeframe = "1d"
leverage = 1.0