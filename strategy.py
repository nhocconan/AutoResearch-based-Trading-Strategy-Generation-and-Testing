#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + 1d RSI filter + volume confirmation
# - Primary signal: 12h Kaufman Adaptive Moving Average (KAMA) - price above KAMA for long, below for short
# - Trend filter: 1d RSI(14) - long when RSI > 50 (bullish bias), short when RSI < 50 (bearish bias)
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: KAMA adapts to market noise, RSI filter ensures higher timeframe momentum alignment

name = "12h_1d_kama_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute 1d RSI(14) for trend filter
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_1d = (100 - (100 / (1 + rs))).fillna(50).values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Pre-compute 12h KAMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama_12h = np.full_like(close_12h, np.nan)
    kama_12h[9] = close_12h[9]  # Seed with first close
    for i in range(10, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc[i] * (close_12h[i] - kama_12h[i-1])
    
    # Align KAMA to primary timeframe (completed 12h bar only)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA OR RSI drops below 50 (lose bullish bias)
            if close[i] < kama_aligned[i] or rsi_14_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above KAMA OR RSI rises above 50 (lose bearish bias)
            if close[i] > kama_aligned[i] or rsi_14_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for KAMA alignment with volume confirmation and RSI filter
            # Long: price above KAMA AND volume regime AND RSI > 50 (bullish bias)
            if (close[i] > kama_aligned[i] and 
                volume_regime[i] and 
                rsi_14_aligned[i] > 50):
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA AND volume regime AND RSI < 50 (bearish bias)
            elif (close[i] < kama_aligned[i] and 
                  volume_regime[i] and 
                  rsi_14_aligned[i] < 50):
                position = -1
                signals[i] = -0.25
    
    return signals