#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily ATR for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio_daily = atr_daily / daily_close
    
    # Align daily ATR ratio to 12h timeframe
    atr_ratio_12h = align_htf_to_ltf(prices, daily, atr_ratio_daily)
    
    # Calculate daily RSI for momentum filter
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_daily = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_12h = align_htf_to_ltf(prices, daily, rsi_daily)
    
    # Calculate daily Bollinger Bands for mean reversion signals
    sma_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 12h timeframe
    upper_band_12h = align_htf_to_ltf(prices, daily, upper_band)
    lower_band_12h = align_htf_to_ltf(prices, daily, lower_band)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_12h[i]) or np.isnan(rsi_12h[i]) or 
            np.isnan(upper_band_12h[i]) or np.isnan(lower_band_12h[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion strategy with volatility and momentum filters
        # Long when price touches lower BB in low volatility + oversold RSI
        if (close[i] <= lower_band_12h[i] and 
            atr_ratio_12h[i] < 0.01 and  # Low volatility filter
            rsi_12h[i] < 30):  # Oversold filter
            signals[i] = 0.25
        # Short when price touches upper BB in low volatility + overbought RSI
        elif (close[i] >= upper_band_12h[i] and 
              atr_ratio_12h[i] < 0.01 and  # Low volatility filter
              rsi_12h[i] > 70):  # Overbought filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_DailyBB_RSI_Volatility_MeanReversion"
timeframe = "12h"
leverage = 1.0