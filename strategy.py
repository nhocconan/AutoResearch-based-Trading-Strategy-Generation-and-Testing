#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour EMA crossover with 12-hour RSI filter and volume confirmation.
# Long when: EMA9 crosses above EMA21 AND RSI12h > 50 AND volume > 1.2x 20-period average
# Short when: EMA9 crosses below EMA21 AND RSI12h < 50 AND volume > 1.2x 20-period average
# Exit when: EMA crossover reverses OR RSI12h crosses 50 in opposite direction
# Uses 12h RSI as trend filter to avoid counter-trend trades. EMA9/21 for entry timing.
# Volume confirms momentum. Designed for 12-25 trades/year per symbol.
name = "6h_EMA9x21_RSI12h_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMAs for entry signals
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Load 12h data ONCE before loop (Rule 1 compliance)
    df_12h = get_htf_data(prices, '12h')
    # Calculate RSI on 12h closes
    rsi_period = 14
    delta = pd.Series(df_12h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.fillna(50).values  # Neutral when undefined
    # Align 12h RSI to 6h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Wait for EMA21 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(rsi_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # EMA crossover signals
        ema9_above = ema9[i] > ema21[i]
        ema9_above_prev = ema9[i-1] > ema21[i-1]
        ema_cross_up = ema9_above and not ema9_above_prev
        ema_cross_down = not ema9_above and ema9_above_prev
        
        rsi = rsi_12h_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: EMA9 crosses above EMA21 AND RSI12h > 50 AND volume spike
            if ema_cross_up and rsi > 50 and vol > 1.2 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: EMA9 crosses below EMA21 AND RSI12h < 50 AND volume spike
            elif ema_cross_down and rsi < 50 and vol > 1.2 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA9 crosses below EMA21 OR RSI12h drops below 50
            if ema_cross_down or rsi < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA9 crosses above EMA21 OR RSI12h rises above 50
            if ema_cross_up or rsi > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals