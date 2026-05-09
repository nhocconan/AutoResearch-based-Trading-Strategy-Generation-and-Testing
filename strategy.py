#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI4_Div_Liquidity_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and liquidity analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily RSI(4) for divergence detection
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = loss.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi4_1d = (100 - (100 / (1 + rs))).values
    rsi4_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi4_1d)
    
    # Calculate daily volume average for liquidity filter
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma10_1d = vol_1d.rolling(window=10, min_periods=10).mean().values
    vol_ma10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma10_1d)
    
    # Calculate 4h RSI(4) for entry timing
    delta_4h = pd.Series(close).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs_4h = avg_gain_4h / avg_loss_4h.replace(0, 1e-10)
    rsi4_4h = (100 - (100 / (1 + rs_4h))).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi4_1d_aligned[i]) or 
            np.isnan(vol_ma10_1d_aligned[i]) or np.isnan(rsi4_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Liquidity filter: current 4h volume > 1.5x daily average volume / 6
        vol_liquidity = volume[i] > 1.5 * (vol_ma10_1d_aligned[i] / 6)
        
        if position == 0:
            # Long: RSI(4) bullish divergence + oversold + liquidity + above daily EMA
            if (rsi4_4h[i] < 30 and  # oversold on 4h
                rsi4_1d_aligned[i] < rsi4_1d_aligned[i-1] and  # daily RSI falling
                low[i] < low[i-1] and  # price making lower low
                rsi4_1d_aligned[i] > rsi4_1d_aligned[i-1] and  # but RSI making higher low (bullish div)
                vol_liquidity and
                close[i] > ema34_1d_aligned[i]):  # above daily trend
                signals[i] = 0.25
                position = 1
            # Short: RSI(4) bearish divergence + overbought + liquidity + below daily EMA
            elif (rsi4_4h[i] > 70 and  # overbought on 4h
                  rsi4_1d_aligned[i] > rsi4_1d_aligned[i-1] and  # daily RSI rising
                  high[i] > high[i-1] and  # price making higher high
                  rsi4_1d_aligned[i] < rsi4_1d_aligned[i-1] and  # but RSI making lower high (bearish div)
                  vol_liquidity and
                  close[i] < ema34_1d_aligned[i]):  # below daily trend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI(4) overbought or trend change
            if rsi4_4h[i] > 70 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI(4) oversold or trend change
            if rsi4_4h[i] < 30 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals