#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI4_Divergence_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    close_w = df_w['close'].values
    
    # Weekly EMA(34) for trend filter
    close_w_series = pd.Series(close_w)
    ema34_w = close_w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_w_aligned = align_htf_to_ltf(prices, df_w, ema34_w)
    
    # RSI(4) on 6h close
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = loss.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # RSI divergence detection: look for bearish/bullish divergence over last 5 bars
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    def check_divergence(prices, rsi_vals, lookback=5):
        bullish_div = np.zeros(len(prices), dtype=bool)
        bearish_div = np.zeros(len(prices), dtype=bool)
        for i in range(lookback, len(prices)):
            # Price lows and highs in lookback window
            price_low = np.min(prices[i-lookback:i+1])
            price_high = np.max(prices[i-lookback:i+1])
            rsi_low = np.min(rsi_vals[i-lookback:i+1])
            rsi_high = np.max(rsi_vals[i-lookback:i+1])
            
            # Current price vs past
            if prices[i] == price_low and rsi_vals[i] > rsi_low:
                bullish_div[i] = True
            if prices[i] == price_high and rsi_vals[i] < rsi_high:
                bearish_div[i] = True
        return bullish_div, bearish_div
    
    bullish_div, bearish_div = check_divergence(close, rsi_values, lookback=5)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_w_aligned[i]) or np.isnan(vol_ma20[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Bullish RSI divergence + above weekly EMA trend + volume
            if bullish_div[i] and vol_ok and close[i] > ema34_w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish RSI divergence + below weekly EMA trend + volume
            elif bearish_div[i] and vol_ok and close[i] < ema34_w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish RSI divergence or price below weekly EMA
            if bearish_div[i] or close[i] < ema34_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish RSI divergence or price above weekly EMA
            if bullish_div[i] or close[i] > ema34_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals