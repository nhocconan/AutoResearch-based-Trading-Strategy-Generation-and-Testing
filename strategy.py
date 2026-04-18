#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d RSI trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) with RSI(14) > 50 and volume > 1.5x 24-bar average.
# Short when price breaks below lower BB(20,2) with RSI(14) < 50 and volume > 1.5x 24-bar average.
# Exit when price crosses back inside Bollinger Bands.
# Uses 1d RSI to filter counter-trend trades, volume to confirm conviction, and Bollinger Bands for volatility-based entry.
# Designed for ~30-50 trades/year with low turnover to minimize fee drag.
name = "4h_BollingerBreakout_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for RSI trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # RSI(14) on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Bollinger Bands(20,2) on 4h close
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 4h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        rsi_val = rsi_14_1d_aligned[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above upper BB with RSI > 50 and volume
            if close_val > upper and rsi_val > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB with RSI < 50 and volume
            elif close_val < lower and rsi_val < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back inside BB (below upper)
            if close_val < upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back inside BB (above lower)
            if close_val > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals