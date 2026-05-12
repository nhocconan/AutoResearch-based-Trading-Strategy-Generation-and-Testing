#!/usr/bin/env python3
# 4h_RSI_Divergence_With_Volume_Trend_Filter
# Hypothesis: Bullish/bearish RSI divergence on 1d timeframe combined with 4h volume confirmation and EMA trend filter captures high-probability reversals in both bull and bear markets. Divergence signals exhaustion of momentum, while volume and trend filters ensure alignment with institutional flow. Targets 20-40 trades/year to minimize fee drag.

name = "4h_RSI_Divergence_With_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate 14-period RSI on 1d
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values

    # Calculate 4-period RSI for divergence detection
    delta_short = pd.Series(close_1d).diff()
    gain_short = delta_short.clip(lower=0)
    loss_short = -delta_short.clip(upper=0)
    avg_gain_short = gain_short.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss_short = loss_short.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs_short = avg_gain_short / avg_loss_short
    rsi_4_1d = 100 - (100 / (1 + rs_short))
    rsi_4_1d_values = rsi_4_1d.values

    # Detect bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    def detect_divergence(price, rsi, lookback=10):
        n_len = len(price)
        bullish_div = np.zeros(n_len, dtype=bool)
        bearish_div = np.zeros(n_len, dtype=bool)
        
        for i in range(lookback, n_len):
            # Look for local lows in price and RSI
            price_low = np.argmin(price[i-lookback:i+1]) + i - lookback
            rsi_low = np.argmin(rsi[i-lookback:i+1]) + i - lookback
            
            # Look for local highs in price and RSI
            price_high = np.argmax(price[i-lookback:i+1]) + i - lookback
            rsi_high = np.argmax(rsi[i-lookback:i+1]) + i - lookback
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (price[i] < price[price_low] and 
                rsi[i] > rsi[rsi_low] and
                price_low != i and rsi_low != i):
                bullish_div[i] = True
                
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (price[i] > price[price_high] and 
                rsi[i] < rsi[rsi_high] and
                price_high != i and rsi_high != i):
                bearish_div[i] = True
                
        return bullish_div, bearish_div

    # Detect divergences on 1d data
    bullish_div_1d, bearish_div_1d = detect_divergence(close_1d, rsi_1d_values, lookback=10)
    
    # Also check short-term RSI for confirmation
    bullish_div_4_1d, bearish_div_4_1d = detect_divergence(close_1d, rsi_4_1d_values, lookback=6)
    
    # Combine signals: require both RSIs to show divergence
    bullish_div_final = bullish_div_1d & bullish_div_4_1d
    bearish_div_final = bearish_div_1d & bearish_div_4_1d

    # Align divergence signals to 4h timeframe
    bullish_div_aligned = align_htf_to_ltf(prices, df_1d, bullish_div_final.astype(float))
    bearish_div_aligned = align_htf_to_ltf(prices, df_1d, bearish_div_final.astype(float))

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(bullish_div_aligned[i]) or np.isnan(bearish_div_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish divergence + price above EMA50 (uptrend) + volume confirmation
            if (bullish_div_aligned[i] > 0.5 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + price below EMA50 (downtrend) + volume confirmation
            elif (bearish_div_aligned[i] > 0.5 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or price closes below EMA50
            if (bearish_div_aligned[i] > 0.5 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence or price closes above EMA50
            if (bullish_div_aligned[i] > 0.5 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals