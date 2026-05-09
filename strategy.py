# 1D_KAMA_Trend_Stochastic_RSI_Confirmation
# Hypothesis: On 1d timeframe, enter long when KAMA indicates uptrend and StochRSI is oversold (<0.2), enter short when KAMA indicates downtrend and StochRSI is overbought (>0.8).
# Uses 1w trend filter to avoid counter-trend trades and KAMA for smooth trend following.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years).

name = "1D_KAMA_Trend_Stochastic_RSI_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA and StochRSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA calculation
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[length:] = change / (volatility + 1e-10)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Stochastic RSI
    def stoch_rsi(close, length=14, rsi_length=14, stoch_length=14):
        delta = np.diff(close)
        up = np.clip(delta, 0, None)
        down = np.clip(-delta, 0, None)
        ma_up = np.zeros_like(close)
        ma_down = np.zeros_like(close)
        ma_up[rsi_length] = np.mean(up[:rsi_length])
        ma_down[rsi_length] = np.mean(down[:rsi_length])
        for i in range(rsi_length+1, len(close)):
            ma_up[i] = (ma_up[i-1] * (rsi_length-1) + up[i-1]) / rsi_length
            ma_down[i] = (ma_down[i-1] * (rsi_length-1) + down[i-1]) / rsi_length
        rsi = np.zeros_like(close)
        rsi[rsi_length:] = 100 * ma_up[rsi_length:] / (ma_up[rsi_length:] + ma_down[rsi_length:] + 1e-10)
        # Stochastic of RSI
        stoch_rsi = np.zeros_like(close)
        for i in range(stoch_length-1, len(rsi)):
            min_rsi = np.min(rsi[i-stoch_length+1:i+1])
            max_rsi = np.max(rsi[i-stoch_length+1:i+1])
            if max_rsi - min_rsi > 1e-10:
                stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi)
            else:
                stoch_rsi[i] = 0.5
        return stoch_rsi
    
    # Calculate indicators
    kama = calculate_kama(close_1d, length=10, fast=2, slow=30)
    stoch_rsi_val = stoch_rsi(close_1d, length=14, rsi_length=14, stoch_length=14)
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1w = close_1w > ema_20_1w
    
    # Align to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    stoch_rsi_aligned = align_htf_to_ltf(prices, df_1d, stoch_rsi_val)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(stoch_rsi_aligned[i]) or np.isnan(trend_up_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA uptrend + StochRSI oversold + 1w uptrend
            if close_1d[i] > kama_aligned[i] and stoch_rsi_aligned[i] < 0.2 and trend_up_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA downtrend + StochRSI overbought + 1w downtrend
            elif close_1d[i] < kama_aligned[i] and stoch_rsi_aligned[i] > 0.8 and not trend_up_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA downtrend or StochRSI overbought
            if close_1d[i] < kama_aligned[i] or stoch_rsi_aligned[i] > 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA uptrend or StochRSI oversold
            if close_1d[i] > kama_aligned[i] or stoch_rsi_aligned[i] < 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals