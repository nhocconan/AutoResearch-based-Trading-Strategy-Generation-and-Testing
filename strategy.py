#!/usr/bin/env python3
# 4h_12h_rsi_divergence_volume_v1
# Hypothesis: Use 12h RSI divergence (bullish/bearish) for early trend reversal signals, confirmed by 4h price action and volume spikes. Works in bull markets (catches reversals from oversold) and bear markets (catches reversals from overbought). Target: 25-40 trades/year per symbol (100-160 total over 4 years) by requiring RSI divergence + price confirmation + volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_rsi_divergence_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for RSI divergence
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h RSI(14)
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:14] = np.nan  # First 14 values invalid
    
    # Calculate 12h price swing points for divergence detection
    # Simple swing high/low: look for local extrema over 3-period window
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Find swing highs (local maxima)
    swing_high = np.zeros_like(high_12h, dtype=bool)
    swing_low = np.zeros_like(low_12h, dtype=bool)
    for i in range(1, len(high_12h)-1):
        if high_12h[i] >= high_12h[i-1] and high_12h[i] >= high_12h[i+1]:
            swing_high[i] = True
        if low_12h[i] <= low_12h[i-1] and low_12h[i] <= low_12h[i+1]:
            swing_low[i] = True
    
    # Detect bullish RSI divergence: price makes lower low, RSI makes higher low
    bullish_div = np.zeros_like(rsi, dtype=bool)
    bearish_div = np.zeros_like(rsi, dtype=bool)
    
    # Track recent swing points
    last_price_low = np.nan
    last_price_low_idx = -1
    last_rsi_low = np.nan
    last_price_high = np.nan
    last_price_high_idx = -1
    last_rsi_high = np.nan
    
    for i in range(len(rsi)):
        if np.isnan(rsi[i]):
            continue
            
        # Update swing points
        if swing_low[i]:
            last_price_low = low_12h[i]
            last_price_low_idx = i
            last_rsi_low = rsi[i]
        if swing_high[i]:
            last_price_high = high_12h[i]
            last_price_high_idx = i
            last_rsi_high = rsi[i]
            
        # Check for bullish divergence (need at least one prior swing low)
        if not np.isnan(last_price_low) and i > last_price_low_idx:
            if low_12h[i] < last_price_low and rsi[i] > last_rsi_low:
                bullish_div[i] = True
                
        # Check for bearish divergence (need at least one prior swing high)
        if not np.isnan(last_price_high) and i > last_price_high_idx:
            if high_12h[i] > last_price_high and rsi[i] < last_rsi_high:
                bearish_div[i] = True
    
    # Align divergence signals to 4h timeframe
    bullish_div_aligned = align_htf_to_ltf(prices, df_12h, bullish_div.astype(float))
    bearish_div_aligned = align_htf_to_ltf(prices, df_12h, bearish_div.astype(float))
    
    # Volume confirmation: volume > 1.5x average of last 96 periods (1 day in 4h)
    vol_ma = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(bullish_div_aligned[i]) or np.isnan(bearish_div_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bearish divergence appears or price closes below 4-period EMA
            ema_4 = pd.Series(close).ewm(span=4, adjust=False).mean().values
            if bearish_div_aligned[i] > 0.5 or close[i] < ema_4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: bullish divergence appears or price closes above 4-period EMA
            ema_4 = pd.Series(close).ewm(span=4, adjust=False).mean().values
            if bullish_div_aligned[i] > 0.5 or close[i] > ema_4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: bullish divergence with volume confirmation
            if bullish_div_aligned[i] > 0.5 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish divergence with volume confirmation
            elif bearish_div_aligned[i] > 0.5 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals