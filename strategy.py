#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 3-period RSI + 1d MACD histogram + volume spike
# - RSI(3) identifies short-term extremes (oversold <20, overbought >80) for mean reversion
# - MACD histogram on 1d confirms momentum direction (bullish when hist > 0 and rising)
# - Volume spike (>2x 20-period average) filters weak moves
# - Works in both bull/bear: mean reversion in ranges, momentum in trends
# - Target: 15-25 trades/year per symbol (~60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for MACD
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d MACD components
    close_1d = pd.Series(df_1d['close'].values)
    ema12 = close_1d.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close_1d.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - signal_line
    macd_hist_values = macd_hist.values
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist_values)
    
    # RSI(3) on 12h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(macd_hist_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: RSI < 20 (oversold) AND MACD hist > 0 and rising + volume
        if (rsi_values[i] < 20 and 
            macd_hist_aligned[i] > 0 and 
            macd_hist_aligned[i] > macd_hist_aligned[i-1] and
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: RSI > 80 (overbought) AND MACD hist < 0 and falling + volume
        elif (rsi_values[i] > 80 and 
              macd_hist_aligned[i] < 0 and 
              macd_hist_aligned[i] < macd_hist_aligned[i-1] and
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_RSI3_MACDHist_Volume"
timeframe = "12h"
leverage = 1.0