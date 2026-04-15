#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly ATR for volatility filter
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    tr = np.maximum(weekly_high - weekly_low,
                    np.maximum(np.abs(weekly_high - weekly_close_prev),
                               np.abs(weekly_low - weekly_close_prev)))
    atr_weekly = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio_weekly = atr_weekly / weekly_close
    
    # Align weekly ATR ratio to 4h timeframe
    atr_ratio_4h = align_htf_to_ltf(prices, weekly, atr_ratio_weekly)
    
    # Calculate weekly RSI for momentum filter
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_weekly = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 4h timeframe
    rsi_4h = align_htf_to_ltf(prices, weekly, rsi_weekly)
    
    # Calculate weekly Bollinger Bands for mean reversion signals
    sma_20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 4h timeframe
    upper_band_4h = align_htf_to_ltf(prices, weekly, upper_band)
    lower_band_4h = align_htf_to_ltf(prices, weekly, lower_band)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_4h[i]) or np.isnan(rsi_4h[i]) or 
            np.isnan(upper_band_4h[i]) or np.isnan(lower_band_4h[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion strategy with volatility and momentum filters
        # Long when price touches lower BB in low volatility + oversold RSI
        if (close[i] <= lower_band_4h[i] and 
            atr_ratio_4h[i] < 0.015 and  # Low volatility filter
            rsi_4h[i] < 30):  # Oversold filter
            signals[i] = 0.25
        # Short when price touches upper BB in low volatility + overbought RSI
        elif (close[i] >= upper_band_4h[i] and 
              atr_ratio_4h[i] < 0.015 and  # Low volatility filter
              rsi_4h[i] > 70):  # Overbought filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_WeeklyBB_RSI_Volatility_MeanReversion"
timeframe = "4h"
leverage = 1.0