#!/usr/bin/env python3
# 6h_1w_1d_rsi_divergence_volume_confirm_v1
# Hypothesis: 6h RSI divergence with volume confirmation and 1w trend filter.
# Bullish divergence: price makes lower low, RSI makes higher low -> long if 1w uptrend.
# Bearish divergence: price makes higher high, RSI makes lower high -> short if 1w downtrend.
# Volume confirmation: current volume > 1.5 * 20-period average volume.
# Designed for 15-30 trades/year on 6h to avoid fee drag. Works in bull/bear via 1w trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_rsi_divergence_volume_confirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 14)  # Ensure RSI and volume average are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # 1w trend filter
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish divergence or volume confirmation lost
            # Check for bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = False
            if i >= 5:
                # Look for recent highs in price and RSI
                price_high_idx = np.argmax(high[i-4:i+1]) + i - 4
                rsi_high_idx = np.argmax(rsi[i-4:i+1]) + i - 4
                if price_high_idx == i-4 and rsi_high_idx == i:  # current bar is high for both
                    if high[i] > high[i-4] and rsi[i] < rsi[i-4]:
                        bearish_div = True
            if bearish_div or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish divergence or volume confirmation lost
            # Check for bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = False
            if i >= 5:
                # Look for recent lows in price and RSI
                price_low_idx = np.argmin(low[i-4:i+1]) + i - 4
                rsi_low_idx = np.argmin(rsi[i-4:i+1]) + i - 4
                if price_low_idx == i-4 and rsi_low_idx == i:  # current bar is low for both
                    if low[i] < low[i-4] and rsi[i] > rsi[i-4]:
                        bullish_div = True
            if bullish_div or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = False
            if i >= 5:
                # Look for recent lows in price and RSI
                price_low_idx = np.argmin(low[i-4:i+1]) + i - 4
                rsi_low_idx = np.argmin(rsi[i-4:i+1]) + i - 4
                if price_low_idx == i-4 and rsi_low_idx == i:  # current bar is low for both
                    if low[i] < low[i-4] and rsi[i] > rsi[i-4]:
                        bullish_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = False
            if i >= 5:
                # Look for recent highs in price and RSI
                price_high_idx = np.argmax(high[i-4:i+1]) + i - 4
                rsi_high_idx = np.argmax(rsi[i-4:i+1]) + i - 4
                if price_high_idx == i-4 and rsi_high_idx == i:  # current bar is high for both
                    if high[i] > high[i-4] and rsi[i] < rsi[i-4]:
                        bearish_div = True
            
            # Long entry: bullish divergence + volume confirmation + 1w uptrend
            if bullish_div and vol_confirm and uptrend_1w:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish divergence + volume confirmation + 1w downtrend
            elif bearish_div and vol_confirm and downtrend_1w:
                position = -1
                signals[i] = -0.25
    
    return signals