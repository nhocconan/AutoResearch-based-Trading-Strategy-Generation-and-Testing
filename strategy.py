#/usr/bin/env python3
"""
4h_RSI_Trend_Filter_With_Volume_Spike
Hypothesis: RSI(14) crossing above 55 with volume confirmation and aligned 1d trend (close > EMA50) signals bullish continuation. RSI crossing below 45 with volume confirmation and aligned 1d trend (close < EMA50) signals bearish continuation. Uses 25% position size to limit trade frequency (~25-40/year) and minimize fee drag in 4-hour bars.
"""

name = "4h_RSI_Trend_Filter_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # neutral before warmup
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        if position == 0:
            # LONG: RSI crosses above 55, volume confirmation, price above 1d EMA50 (uptrend)
            if (rsi_values[i] > 55 and rsi_values[i-1] <= 55 and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below 45, volume confirmation, price below 1d EMA50 (downtrend)
            elif (rsi_values[i] < 45 and rsi_values[i-1] >= 45 and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below 45 OR volume drops
            if (rsi_values[i] < 45 and rsi_values[i-1] >= 45) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above 55 OR volume drops
            if (rsi_values[i] > 55 and rsi_values[i-1] <= 55) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals