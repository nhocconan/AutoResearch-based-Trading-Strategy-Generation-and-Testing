#!/usr/bin/env python3
# 1h_rsi_divergence_4h_trend_volume_v1
# Hypothesis: 1h RSI divergence with 4h trend filter and volume confirmation captures reversal entries in both bull and bear markets.
# Uses 4h EMA50 for trend direction, 1h RSI(14) for momentum, and volume spike for confirmation.
# Target: 15-37 trades/year by requiring confluence of trend, momentum divergence, and volume spike.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_divergence_4h_trend_volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 1h price action: higher low for bullish divergence, lower high for bearish
    # We'll use simple price comparison with lookback
    price_low = np.minimum.accumulate(low)
    price_high = np.maximum.accumulate(high)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(price_low[i]) or np.isnan(price_high[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 60 or price < 4h EMA50
            if (rsi[i] > 60) or (close[i] < ema_50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 or price > 4h EMA50
            if (rsi[i] < 40) or (close[i] > ema_50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Bullish divergence: price making higher low, RSI making lower low
            bullish_div = (low[i] > price_low[i-1]) and (rsi[i] < rsi[i-1])
            # Bearish divergence: price making lower high, RSI making higher high
            bearish_div = (high[i] < price_high[i-1]) and (rsi[i] > rsi[i-1])
            
            # Long entry: bullish divergence + volume spike + price > 4h EMA50 (uptrend)
            if bullish_div and volume_spike[i] and (close[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: bearish divergence + volume spike + price < 4h EMA50 (downtrend)
            elif bearish_div and volume_spike[i] and (close[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals