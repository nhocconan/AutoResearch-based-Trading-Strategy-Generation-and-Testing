# 2025-01-20: Strategy for 4h BTC/ETH/SOL - RSI Divergence with Volume Confirmation
# Hypothesis: RSI divergence (bullish/bearish) combined with volume spikes captures 
# turning points in both bull and bear markets. Volume confirms institutional participation 
# at reversal points. RSI divergence identifies exhaustion moves before price reverses.
# Timeframe: 4h balances signal quality and trade frequency. Uses 14-period RSI.
# Expects 20-35 trades/year per symbol (80-140 total over 4 years), within optimal range.

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
    
    # RSI calculation (14 periods)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to RMA)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 gains
    avg_loss[13] = np.mean(loss[1:14])  # First average of first 14 losses
    
    for i in range(14, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan  # Not enough data for first 13 periods
    
    # Volume confirmation: 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for RSI and volume MA
    start = max(20, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(rsi[i-1]) or
            np.isnan(rsi[i-2]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Bullish RSI divergence: price makes lower low, RSI makes higher low
            bullish_div = (low[i] < low[i-1] and low[i-1] < low[i-2] and 
                          rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2])
            # Bearish RSI divergence: price makes higher high, RSI makes lower high
            bearish_div = (high[i] > high[i-1] and high[i-1] > high[i-2] and 
                          rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2])
            
            if bullish_div and volume_confirmed:
                position = 1
                signals[i] = position_size
            elif bearish_div and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI becomes overbought (>70) or bearish divergence
            exit_signal = (rsi[i] > 70 or 
                          (high[i] > high[i-1] and high[i-1] > high[i-2] and 
                           rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]))
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI becomes oversold (<30) or bullish divergence
            exit_signal = (rsi[i] < 30 or 
                          (low[i] < low[i-1] and low[i-1] < low[i-2] and 
                           rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]))
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_RSI_Divergence_Volume_v1"
timeframe = "4h"
leverage = 1.0