#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA trend filter + RSI mean reversion + volume spike
# In bull markets: 1w EMA up + RSI < 30 + volume spike = long
# In bear markets: 1w EMA down + RSI > 70 + volume spike = short
# Volume spike filters for conviction, reducing false signals
# Low trade frequency expected (<25/year) to minimize fee drag
# Works in both regimes by adapting to 1w trend

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: 1w uptrend + RSI oversold + volume spike
        if (close[i] > ema_50_1w_aligned[i] and  # 1w EMA trend filter (bullish)
            rsi_values[i] < 30 and               # RSI oversold
            volume_ratio[i] > 2.0):              # Volume spike (conviction)
            signals[i] = 0.25
            
        # Short conditions: 1w downtrend + RSI overbought + volume spike
        elif (close[i] < ema_50_1w_aligned[i] and  # 1w EMA trend filter (bearish)
              rsi_values[i] > 70 and               # RSI overbought
              volume_ratio[i] > 2.0):              # Volume spike (conviction)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_EMA50_RSI_Volume_Spike_v1"
timeframe = "1d"
leverage = 1.0