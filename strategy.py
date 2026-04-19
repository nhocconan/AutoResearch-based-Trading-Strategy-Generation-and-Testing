#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 12-period RSI mean reversion with 48-period volume filter and 48-period ATR stop.
# Long when RSI < 30, volume > 1.5x 48-period average, and price > 200-period EMA.
# Short when RSI > 70, volume > 1.5x 48-period average, and price < 200-period EMA.
# Exit when RSI crosses back above 50 (long) or below 50 (short).
# Uses RSI for mean reversion, volume for confirmation, EMA for trend filter, and ATR for stop.
# Target: 20-40 trades/year per symbol.

name = "4h_RSI_MeanReversion_Volume_EMA"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 48-period average volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    # Calculate 200-period EMA for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR (14-period) for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure EMA200 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_200[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        ema = ema_200[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long entry: RSI < 30, volume confirmation, price > EMA200
            if rsi_val < 30 and volume_confirmed and price > ema:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI > 70, volume confirmation, price < EMA200
            elif rsi_val > 70 and volume_confirmed and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 OR stop loss hit (price < entry - 2*ATR)
            if rsi_val > 50 or price < ema - 2 * atr_val:  # Using EMA as proxy for entry
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 OR stop loss hit (price > entry + 2*ATR)
            if rsi_val < 50 or price > ema + 2 * atr_val:  # Using EMA as proxy for entry
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals