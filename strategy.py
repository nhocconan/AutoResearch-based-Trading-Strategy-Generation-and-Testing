# 4h_RSI_MeanReversion_4hTrend_Volume
# Strategy uses 4-hour RSI with mean reversion logic and 4-hour trend filter.
# Enters long when RSI < 30 and price above 4h EMA50 with volume confirmation.
# Enters short when RSI > 70 and price below 4h EMA50 with volume confirmation.
# Exits when RSI crosses back to neutral (40-60) or trend reverses.
# Designed to work in both bull and bear markets by combining mean reversion with trend filter.
# Target: 20-50 trades/year to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI_MeanReversion_4hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan  # Not enough data
    
    # EMA50 on 4h close
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        ema50_val = ema50[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: RSI oversold + price above EMA50 + volume spike
            if rsi_val < 30 and close[i] > ema50_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought + price below EMA50 + volume spike
            elif rsi_val > 70 and close[i] < ema50_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or price below EMA50
            if rsi_val > 40 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or price above EMA50
            if rsi_val < 60 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals