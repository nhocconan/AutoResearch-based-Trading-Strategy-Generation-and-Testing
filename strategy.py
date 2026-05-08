#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_RSI_Divergence_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h RSI for divergence detection
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0).values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA21 for trend filter
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # 12h EMA21 for exit
    ema_21_12h = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(ema_21_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # RSI divergence detection (lookback 3 periods)
        if i >= 3:
            price_higher_high = close[i] > close[i-3] and close[i-1] > close[i-4] and close[i-2] > close[i-5]
            price_lower_low = close[i] < close[i-3] and close[i-1] < close[i-4] and close[i-2] < close[i-5]
            rsi_lower_high = rsi[i] < rsi[i-3] and rsi[i-1] < rsi[i-4] and rsi[i-2] < rsi[i-5]
            rsi_higher_low = rsi[i] > rsi[i-3] and rsi[i-1] > rsi[i-4] and rsi[i-2] > rsi[i-5]
            
            bullish_div = price_lower_low and rsi_higher_low
            bearish_div = price_higher_high and rsi_lower_high
        else:
            bullish_div = False
            bearish_div = False
        
        if position == 0:
            # Long: bullish RSI divergence + uptrend + volume spike
            long_cond = bullish_div and (close[i] > ema_21_1d_aligned[i]) and volume_spike[i]
            # Short: bearish RSI divergence + downtrend + volume spike
            short_cond = bearish_div and (close[i] < ema_21_1d_aligned[i]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish RSI divergence or price below EMA21
            if bearish_div or (close[i] < ema_21_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish RSI divergence or price above EMA21
            if bullish_div or (close[i] > ema_21_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals