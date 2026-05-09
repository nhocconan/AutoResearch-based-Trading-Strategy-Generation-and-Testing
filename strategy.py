#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d RSI mean reversion
# Uses Choppiness Index (14) to detect ranging markets (CHOP > 61.8) and trending markets (CHOP < 38.2)
# In ranging markets: RSI < 30 long, RSI > 70 short (mean reversion)
# In trending markets: price > EMA50 long, price < EMA50 short (trend following)
# Combines regime detection with appropriate strategy per regime for robust performance in bull/bear markets.
name = "4h_Choppiness_RSI_EMA50_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Choppiness Index calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate True Range
    high_low = df_4h['high'] - df_4h['low']
    high_close = np.abs(df_4h['high'] - df_4h['close'].shift(1))
    low_close = np.abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_4h['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_4h['low']).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    
    # Align Choppiness Index to 4h (already on 4h timeframe)
    chop_4h = chop  # No alignment needed as both are 4h
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d
    delta = pd.Series(df_1d['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h
    rsi_14_4h = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Get 1d data for EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and RSI calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_4h[i]) or np.isnan(rsi_14_4h[i]) or np.isnan(ema_50_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_4h[i]
        rsi_val = rsi_14_4h[i]
        close_val = close[i]
        ema_val = ema_50_4h[i]
        
        # Regime detection
        ranging = chop_val > 61.8  # Chop > 61.8 = ranging market
        trending = chop_val < 38.2  # Chop < 38.2 = trending market
        
        if position == 0:
            # In ranging market: mean reversion with RSI
            if ranging:
                if rsi_val < 30:  # Oversold - long
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70:  # Overbought - short
                    signals[i] = -0.25
                    position = -1
            # In trending market: trend following with EMA
            elif trending:
                if close_val > ema_val:  # Price above EMA - long
                    signals[i] = 0.25
                    position = 1
                elif close_val < ema_val:  # Price below EMA - short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: opposite signal based on regime
            if ranging:
                if rsi_val > 50:  # Exit when RSI returns to neutral
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # trending
                if close_val < ema_val:  # Exit when price crosses below EMA
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short: opposite signal based on regime
            if ranging:
                if rsi_val < 50:  # Exit when RSI returns to neutral
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # trending
                if close_val > ema_val:  # Exit when price crosses above EMA
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals