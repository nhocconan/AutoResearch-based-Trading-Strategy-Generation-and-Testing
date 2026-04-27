#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d RSI divergence filter and volume confirmation.
# Long when KAMA turns up and RSI shows bullish divergence (higher low in RSI with lower low in price).
# Short when KAMA turns down and RSI shows bearish divergence (lower high in RSI with higher high in price).
# Exit when KAMA reverses direction.
# Uses 1d timeframe for RSI divergence to capture medium-term momentum shifts.
# Target: 20-40 trades/year to minimize fee drift while capturing sustained trends.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI divergence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day RSI(14) for divergence detection
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA on 4h timeframe
    kama_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Efficiency ratio
    change = np.abs(np.diff(close, kama_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[kama_period:] = change[kama_period:] / volatility[kama_period:]
    er[:kama_period] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[kama_period] = close[kama_period]
    for i in range(kama_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: KAMA turning up and bullish RSI divergence
        kama_up = kama[i] > kama[i-1]
        if i >= 2:
            price_lower_low = low[i] < low[i-2] and low[i-1] < low[i-3]
            rsi_higher_low = rsi_1d_aligned[i] > rsi_1d_aligned[i-2] and rsi_1d_aligned[i-1] > rsi_1d_aligned[i-3]
            bullish_div = price_lower_low and rsi_higher_low
        else:
            bullish_div = False
        
        # Short condition: KAMA turning down and bearish RSI divergence
        kama_down = kama[i] < kama[i-1]
        if i >= 2:
            price_higher_high = high[i] > high[i-2] and high[i-1] > high[i-3]
            rsi_lower_high = rsi_1d_aligned[i] < rsi_1d_aligned[i-2] and rsi_1d_aligned[i-1] < rsi_1d_aligned[i-3]
            bearish_div = price_higher_high and rsi_lower_high
        else:
            bearish_div = False
        
        if kama_up and bullish_div and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        elif kama_down and bearish_div and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions: KAMA reverses direction
        elif position == 1 and kama_down:
            signals[i] = 0.0
            position = 0
        elif position == -1 and kama_up:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_RSIDivergence_VolumeFilter"
timeframe = "4h"
leverage = 1.0