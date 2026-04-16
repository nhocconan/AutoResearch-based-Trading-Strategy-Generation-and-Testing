#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d price action with weekly trend filter.
# Uses 1d price relative to 10-day EMA for short-term momentum.
# Weekly trend determined by price above/below 200-week EMA to filter trades.
# Long only in weekly uptrend when 1d price crosses above 10-day EMA with volume.
# Short only in weekly downtrend when 1d price crosses below 10-day EMA with volume.
# Volume confirmation: current volume > 1.5x 20-day average volume.
# Position size 0.25 to manage drawdown. Designed for low trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA(10) for short-term momentum ===
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Weekly EMA(200) for long-term trend ===
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_10[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema10_val = ema_10[i]
        weekly_ema = ema_200_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below 10-day EMA
            if price < ema10_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above 10-day EMA
            if price > ema10_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price crosses back below 10-day EMA
            if price < ema10_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above 10-day EMA
            if price > ema10_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Weekly uptrend: price above 200-week EMA
            if price > weekly_ema:
                # LONG: price crosses above 10-day EMA with volume
                if price > ema10_val and close[i-1] <= ema10_val and vol_ratio_val > 1.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
            # Weekly downtrend: price below 200-week EMA
            elif price < weekly_ema:
                # SHORT: price crosses below 10-day EMA with volume
                if price < ema10_val and close[i-1] >= ema10_val and vol_ratio_val > 1.5:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_EMA10_WeeklyEMA200_TrendFilter_Volume_v1"
timeframe = "1d"
leverage = 1.0