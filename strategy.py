#!/usr/bin/env python3
"""
1h_RSI_Divergence_4HTrendFilter_v1
Hypothesis: Use 4h EMA for trend direction and 1h RSI divergence for entry.
Long when 4h trend up (price > EMA) and bullish RSI divergence (price makes lower low, RSI makes higher low).
Short when 4h trend down (price < EMA) and bearish RSI divergence (price makes higher high, RSI makes lower high).
Volume confirmation: current volume > 1.3x 20-period average volume.
Session filter: 08-20 UTC to avoid low-liquidity hours.
Position size: 0.20 to limit drawdown.
This combines trend following with momentum reversal signals to work in both bull and bear markets while controlling trade frequency.
"""
name = "1h_RSI_Divergence_4HTrendFilter_v1"
timeframe = "1h"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20)
    close_4h = pd.Series(df_4h['close'])
    ema_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral RSI when no loss
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(30, 20)  # Ensure sufficient warmup for RSI and EMA
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (~8 hours on 1h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Bearish divergence: price makes higher high, RSI makes lower high
            lookback = 5  # look back 5 bars for swing points
            
            if i >= lookback:
                # Find recent swing low in price and RSI
                price_low_idx = i - lookback + np.argmin(low[i-lookback:i+1])
                rsi_low_idx = i - lookback + np.argmin(rsi[i-lookback:i+1])
                
                # Find recent swing high in price and RSI
                price_high_idx = i - lookback + np.argmax(high[i-lookback:i+1])
                rsi_high_idx = i - lookback + np.argmax(rsi[i-lookback:i+1])
                
                bullish_div = (low[price_low_idx] < low[i-lookback] and 
                              rsi[rsi_low_idx] > rsi[i-lookback])
                bearish_div = (high[price_high_idx] > high[i-lookback] and 
                              rsi[rsi_high_idx] < rsi[i-lookback])
                
                # Long: 4h trend up (price > EMA) and bullish RSI divergence
                if (close[i] > ema_4h_aligned[i] and bullish_div and 
                    volume_filter[i]):
                    signals[i] = 0.20
                    position = 1
                    bars_since_exit = 0
                # Short: 4h trend down (price < EMA) and bearish RSI divergence
                elif (close[i] < ema_4h_aligned[i] and bearish_div and 
                      volume_filter[i]):
                    signals[i] = -0.20
                    position = -1
                    bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA side or RSI reaches extreme
            if position == 1 and (close[i] < ema_4h_aligned[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (close[i] > ema_4h_aligned[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals