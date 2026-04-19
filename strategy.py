#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation
# Uses 1h as primary timeframe with 4h EMA trend filter and volume spike (>1.5x average)
# Long when: price > 4h EMA20, RSI(14) > 50, volume confirmed
# Short when: price < 4h EMA20, RSI(14) < 50, volume confirmed
# Volume confirmation ensures institutional participation
# Target: 15-37 trades/year per symbol (~60-150 total over 4 years)

name = "1h_EMA20_RSI_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate RSI(14) on 1h data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need EMA and RSI data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_20_4h_aligned[i]
        rsi_val = rsi_values[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price > 4h EMA20 AND RSI > 50 AND volume confirmed
            if price > ema_trend and rsi_val > 50 and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Enter short: price < 4h EMA20 AND RSI < 50 AND volume confirmed
            elif price < ema_trend and rsi_val < 50 and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when price < 4h EMA20 OR RSI <= 50
            if price < ema_trend or rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price > 4h EMA20 OR RSI >= 50
            if price > ema_trend or rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals