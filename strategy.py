# 6h_RSIOscillator_VolumeTrend
# Hypothesis: 6h RSI oscillator with volume trend confirmation for momentum entries
# RSI < 30/70 with volume trend filter avoids false signals in chop
# Volume trend: 20-period volume EMA rising/falling confirms momentum
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear via volume trend filter that adapts to market conditions
# Volume trend reduces whipsaws by requiring institutional participation

name = "6h_RSIOscillator_VolumeTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    avg_gain = WilderSmooth(gain, 14)
    avg_loss = WilderSmooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume trend: 20-period EMA slope
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    volume_ema_slope = np.diff(volume_ema, prepend=volume_ema[0])
    
    # Price momentum: 5-period ROC
    roc = np.zeros_like(close)
    roc[5:] = (close[5:] - close[:-5]) / close[:-5] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(volume_ema_slope[i]) or 
            np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        # Volume trend: positive slope = rising volume, negative = falling
        volume_trend_up = volume_ema_slope[i] > 0
        volume_trend_down = volume_ema_slope[i] < 0
        
        if position == 0:
            # Long: RSI oversold (<30) with rising volume and positive momentum
            if (rsi[i] < 30 and 
                volume_trend_up and 
                roc[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with falling volume and negative momentum
            elif (rsi[i] > 70 and 
                  volume_trend_down and 
                  roc[i] < 0):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI overbought (>70) or volume trend turns down
            if (rsi[i] > 70) or (not volume_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI oversold (<30) or volume trend turns up
            if (rsi[i] < 30) or (not volume_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals