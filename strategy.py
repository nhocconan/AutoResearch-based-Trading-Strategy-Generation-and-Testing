#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d combined RSI mean reversion with volatility filter and session timing
# RSI on 4h and 1d timeframes with smoothed values for trend context
# Mean reversion entries when RSI is oversold/overbought on both timeframes
# Volatility filter using ATR ratio to avoid low volatility whipsaws
# Session filter (08-20 UTC) to focus on active trading hours
# Conservative position sizing to limit drawdown
# Target: 15-25 trades per year (60-100 total over 4 years) for 1h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h RSI (14-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    
    avg_gain_4h = np.zeros_like(gain_4h)
    avg_loss_4h = np.zeros_like(loss_4h)
    avg_gain_4h[13] = np.mean(gain_4h[:14])
    avg_loss_4h[13] = np.mean(loss_4h[:14])
    for i in range(14, len(gain_4h)):
        avg_gain_4h[i] = (avg_gain_4h[i-1] * 13 + gain_4h[i]) / 14
        avg_loss_4h[i] = (avg_loss_4h[i-1] * 13 + loss_4h[i]) / 14
    
    rs_4h = np.divide(avg_gain_4h, avg_loss_4h, out=np.zeros_like(avg_gain_4h), where=avg_loss_4h!=0)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_smoothed = np.convolve(rsi_4h, np.ones(3)/3, mode='same')
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_smoothed)
    
    # Calculate 1d RSI (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = np.zeros_like(gain_1d)
    avg_loss_1d = np.zeros_like(loss_1d)
    avg_gain_1d[13] = np.mean(gain_1d[:14])
    avg_loss_1d[13] = np.mean(loss_1d[:14])
    for i in range(14, len(gain_1d)):
        avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
        avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    
    rs_1d = np.divide(avg_gain_1d, avg_loss_1d, out=np.zeros_like(avg_gain_1d), where=avg_loss_1d!=0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_smoothed = np.convolve(rsi_1d, np.ones(3)/3, mode='same')
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_smoothed)
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.zeros(n)
    atr[13] = np.mean(tr[:14])
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio (current ATR / 50-period average ATR) for volatility filter
    atr_ma = np.zeros(n)
    for i in range(49, n):
        atr_ma[i] = np.mean(atr[i-49:i])
    atr_ratio = np.divide(atr, atr_ma, out=np.ones_like(atr), where=atr_ma!=0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        rsi_4h_val = rsi_4h_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        vol_ratio = atr_ratio[i]
        
        # Volatility filter: avoid low volatility (ratio < 0.8) and extreme volatility (ratio > 2.0)
        volatility_filter = 0.8 <= vol_ratio <= 2.0
        
        if position == 0:
            # Long: RSI oversold on both timeframes + volatility filter
            if (rsi_4h_val < 30 and rsi_1d_val < 30 and volatility_filter):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought on both timeframes + volatility filter
            elif (rsi_4h_val > 70 and rsi_1d_val > 70 and volatility_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or overbought on either timeframe
            if (rsi_4h_val > 50 or rsi_1d_val > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral or oversold on either timeframe
            if (rsi_4h_val < 50 or rsi_1d_val < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_RSI_MeanReversion_VolatilityFilter"
timeframe = "1h"
leverage = 1.0