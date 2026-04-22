#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly Donchian channels for structure
    donch_high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    donch_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20_1w)
    donch_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20_1w)
    
    # Daily ATR for volatility filter and stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(donch_high_20_1w_aligned[i]) or 
            np.isnan(donch_low_20_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr[i]
        ema200_val = ema200_1w_aligned[i]
        donch_high = donch_high_20_1w_aligned[i]
        donch_low = donch_low_20_1w_aligned[i]
        
        # Trend filter: only trade in direction of weekly EMA200
        bullish_trend = price > ema200_val
        bearish_trend = price < ema200_val
        
        if position == 0:
            # Long: break above weekly Donchian high in bullish trend
            if bullish_trend and price > donch_high:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below weekly Donchian low in bearish trend
            elif bearish_trend and price < donch_low:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: opposite Donchian break or ATR-based stop
            if position == 1:
                # Exit long on break below weekly Donchian low or 2*ATR stop
                if price < donch_low or price < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
            else:  # position == -1
                # Exit short on break above weekly Donchian high or 2*ATR stop
                if price > donch_high or price > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "1d_WeeklyDonchianBreakout_EMA200Trend_v1"
timeframe = "1d"
leverage = 1.0