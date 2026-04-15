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
    
    # Get daily data for context
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    daily_volume = daily['volume'].values
    
    # Calculate daily ATR for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio_daily = atr_daily / daily_close
    
    # Align daily ATR ratio to 4h timeframe
    atr_ratio_4h = align_htf_to_ltf(prices, daily, atr_ratio_daily)
    
    # Calculate daily RSI for momentum filter
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_daily = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 4h timeframe
    rsi_4h = align_htf_to_ltf(prices, daily, rsi_daily)
    
    # Calculate daily volume moving average
    volume_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume MA to 4h timeframe
    volume_ma_4h = align_htf_to_ltf(prices, daily, volume_ma)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_4h[i]) or np.isnan(rsi_4h[i]) or 
            np.isnan(volume_ma_4h[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion strategy with volatility and momentum filters
        # Long when price touches lower daily Bollinger Band in low volatility + oversold RSI
        # Short when price touches upper daily Bollinger Band in low volatility + overbought RSI
        
        # Calculate daily Bollinger Bands
        sma_20 = np.mean(daily_close[max(0, i//24-19):i//24+1]) if i//24 >= 19 else np.nan
        std_20 = np.std(daily_close[max(0, i//24-19):i//24+1]) if i//24 >= 19 else np.nan
        if np.isnan(sma_20) or np.isnan(std_20):
            signals[i] = 0.0
            continue
            
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        
        # Check if price is near Bollinger Bands (within 0.5%)
        near_lower = close[i] <= lower_band * 1.005
        near_upper = close[i] >= upper_band * 0.995
        
        # Entry conditions
        if near_lower and atr_ratio_4h[i] < 0.012 and rsi_4h[i] < 30 and volume[i] > volume_ma_4h[i]:
            signals[i] = 0.25
        elif near_upper and atr_ratio_4h[i] < 0.012 and rsi_4h[i] > 70 and volume[i] > volume_ma_4h[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyBB_RSI_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0