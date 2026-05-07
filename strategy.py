#!/usr/bin/env python3
"""
6h_RSI_Trend_Divergence_Volume
Hypothesis: In trending markets, price makes higher highs/lows while RSI diverges (makes lower highs/higher lows), signaling exhaustion. 
Enter on RSI reversal with volume confirmation and trend alignment from 1d trend filter. Works in bull/bear by using trend filter for direction.
Target: 15-30 trades/year (60-120 over 4 years) with position size 0.25.
"""

name = "6h_RSI_Trend_Divergence_Volume"
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
    
    # Load 1-day data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need 30 periods for RSI and EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA50
        uptrend_regime = close[i] > ema_50_1d_aligned[i]
        downtrend_regime = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low -> long in uptrend
            # Bearish divergence: price makes higher high, RSI makes lower high -> short in downtrend
            
            # Check for bullish divergence (need at least 3 bars back)
            if i >= 3:
                # Price lower low
                price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
                # RSI higher low
                rsi_higher_low = rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]
                bullish_div = price_lower_low and rsi_higher_low
                
                # Price higher high
                price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
                # RSI lower high
                rsi_lower_high = rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]
                bearish_div = price_higher_high and rsi_lower_high
                
                # Long: bullish divergence in uptrend + volume
                if bullish_div and uptrend_regime and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish divergence in downtrend + volume
                elif bearish_div and downtrend_regime and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: RSI crosses above 70 (overbought) or trend changes to downtrend
            if (rsi[i] > 70) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses below 30 (oversold) or trend changes to uptrend
            if (rsi[i] < 30) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals