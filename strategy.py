#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h RSI trend filter and 1h volume-weighted VWAP mean reversion
# 4h RSI > 50 indicates bullish trend bias, < 50 indicates bearish bias
# 1h VWAP deviation identifies mean reversion opportunities within the trend
# Volume filter ensures trades occur during sufficient liquidity
# Designed for 15-35 trades/year to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for RSI trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI (14 periods)
    rsi_len = 14
    close_4h = df_4h['close'].values
    
    # Price changes
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    
    # RSI calculation
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Align 4h RSI to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Calculate 1h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3
    vwap_numerator = (typical_price * volume)
    vwap_denominator = volume
    
    # Cumulative VWAP with reset at session boundaries (optional)
    # Using expanding window for simplicity, but resets daily would be better
    vwap = np.cumsum(vwap_numerator) / np.cumsum(vwap_denominator)
    vwap = np.where(np.cumsum(vwap_denominator) == 0, typical_price, vwap)
    
    # VWAP deviation as percentage
    vwap_dev = (close - vwap) / vwap
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(30, rsi_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if np.isnan(rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h RSI > 50 = bullish bias, < 50 = bearish bias
        bullish_bias = rsi_aligned[i] > 50
        bearish_bias = rsi_aligned[i] < 50
        
        # Mean reversion signals from VWAP deviation
        vwap_dev_current = vwap_dev[i]
        
        # Oversold/overbought thresholds
        oversold = vwap_dev_current < -0.015  # -1.5% below VWAP
        overbought = vwap_dev_current > 0.015  # +1.5% above VWAP
        
        # Volume filter: current volume > 20-period average
        if i >= 20:
            vol_avg = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_avg * 0.5  # At least 50% of average
        else:
            volume_filter = True  # Not enough data, don't filter
        
        if position == 0:
            # Enter long: bullish bias + oversold + volume
            if bullish_bias and oversold and volume_filter:
                position = 1
                signals[i] = position_size
            # Enter short: bearish bias + overbought + volume
            elif bearish_bias and overbought and volume_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or trend weakens
            if vwap_dev_current > -0.005 or rsi_aligned[i] < 45:  # Near VWAP or RSI weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP or trend weakens
            if vwap_dev_current < 0.005 or rsi_aligned[i] > 55:  # Near VWAP or RSI weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hRSI_VWAP_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0