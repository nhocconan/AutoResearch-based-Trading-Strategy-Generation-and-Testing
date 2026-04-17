#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility regime
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]  # First value
    low_close_1d[0] = high_low_1d[0]   # First value
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile rank (252-day lookback for regime)
    atr_percentile = pd.Series(atr_1d).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align ATR percentile to 4h timeframe
    atr_percentile_4h = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 4-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for ATR percentile
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_percentile_4h[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade in low volatility (percentile < 30)
        low_volatility = atr_percentile_4h[i] < 30
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Mean reversion signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: RSI oversold in low volatility with volume
            if (rsi_oversold and low_volatility and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in low volatility with volume
            elif (rsi_overbought and low_volatility and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or volatility increases
            if (rsi[i] >= 50) or (atr_percentile_4h[i] >= 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or volatility increases
            if (rsi[i] <= 50) or (atr_percentile_4h[i] >= 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_MeanReversion_LowVol_Volume"
timeframe = "4h"
leverage = 1.0