#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Heikin_Ashi_Momentum_Divergence"
timeframe = "6h"
leverage = 1.0

def calculate_ha(close, open_, high, low):
    """Calculate Heikin Ashi candles"""
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.zeros_like(close)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, len(close)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([high, low, ha_open, ha_close])
    ha_low = np.minimum.reduce([high, low, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Heikin Ashi and momentum divergence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Heikin Ashi on daily
    ha_open_1d, ha_high_1d, ha_low_1d, ha_close_1d = calculate_ha(
        df_1d['close'].values,
        df_1d['open'].values,
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Momentum: 14-period RSI on daily HA close
    def calculate_rsi(data, period=14):
        if len(data) < period:
            return np.full_like(data, np.nan)
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(ha_close_1d, 14)
    
    # Divergence: price making higher highs but RSI making lower highs (bearish)
    # or price making lower lows but RSI making higher lows (bullish)
    def find_divergence(price, rsi, lookback=10):
        bullish_div = np.zeros_like(price, dtype=bool)
        bearish_div = np.zeros_like(price, dtype=bool)
        for i in range(lookback, len(price)):
            # Bullish divergence: price lower low, RSI higher low
            if (price[i] == np.min(price[i-lookback:i+1]) and 
                rsi[i] == np.max(rsi[i-lookback:i+1])):
                bullish_div[i] = True
            # Bearish divergence: price higher high, RSI lower high
            if (price[i] == np.max(price[i-lookback:i+1]) and 
                rsi[i] == np.min(rsi[i-lookback:i+1])):
                bearish_div[i] = True
        return bullish_div, bearish_div
    
    bull_div_1d, bear_div_1d = find_divergence(ha_close_1d, rsi_1d, 10)
    
    # Align to 6h
    ha_close_1d_aligned = align_htf_to_ltf(prices, df_1d, ha_close_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bull_div_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_div_1d.astype(float))
    bear_div_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_div_1d.astype(float))
    
    # 6h momentum confirmation: RSI(14) on HA close
    ha_open, ha_high, ha_low, ha_close = calculate_ha(close, open_, high, low)
    rsi_6h = calculate_rsi(ha_close, 14)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ha_close_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish divergence on daily AND 6h RSI < 30 (oversold)
            if bull_div_1d_aligned[i] and rsi_6h[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence on daily AND 6h RSI > 70 (overbought)
            elif bear_div_1d_aligned[i] and rsi_6h[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish divergence OR 6h RSI > 70
            if bear_div_1d_aligned[i] or rsi_6h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish divergence OR 6h RSI < 30
            if bull_div_1d_aligned[i] or rsi_6h[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Heikin Ashi smooths price action to filter noise, while divergence between price and momentum
# identifies exhaustion points. Works in both bull and bear markets by catching reversals at extremes.
# Daily HA RSI divergence provides the signal, 6h HA RSI provides entry timing confirmation.
# Target: 50-150 trades over 4 years (12-37/year) with discrete positions to minimize fee drag.