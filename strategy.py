#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 1w/1d RSI divergence + volume confirmation + price action.
# Uses weekly RSI(14) divergence with price to identify exhaustion in trends.
# Confirms with daily RSI(14) oversold/overbought and volume spike.
# Works in bull/bear by fading extremes with momentum divergence.
# Target: 20-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Load 1w data for RSI divergence
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta_w = np.diff(close_1w, prepend=close_1w[0])
    gain_w = np.where(delta_w > 0, delta_w, 0)
    loss_w = np.where(delta_w < 0, -delta_w, 0)
    avg_gain_w = pd.Series(gain_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_w = pd.Series(loss_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_w = avg_gain_w / np.where(avg_loss_w == 0, 1, avg_loss_w)
    rsi_1w = 100 - (100 / (1 + rs_w))
    
    # Align weekly RSI to 4h
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h RSI(14) for confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(rsi_1d[i-1]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_4h[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_1d_val = rsi_1d[i-1]  # Use previous day's RSI (already closed)
        rsi_1w_val = rsi_1w_aligned[i]
        rsi_4h_val = rsi_4h[i]
        atr = atr_14[i]
        vol_ratio_4h = vol_ratio[i]
        
        # Volatility filter: avoid extreme volatility
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_4h > 1.5)
        
        # Price action: look for rejection at extremes
        # Bullish rejection: lower wick > 60% of body
        body = np.abs(close[i] - prices['open'].values[i])
        lower_wick = prices['open'].values[i] - low[i] if close[i] >= prices['open'].values[i] else close[i] - low[i]
        bullish_rejection = (lower_wick > 0.6 * body) if body > 0 else False
        
        # Bearish rejection: upper wick > 60% of body
        upper_wick = high[i] - prices['open'].values[i] if close[i] >= prices['open'].values[i] else high[i] - close[i]
        bearish_rejection = (upper_wick > 0.6 * body) if body > 0 else False
        
        if position == 0:
            # Long setup: RSI divergence + oversold + bullish rejection
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Simplified: RSI oversold and turning up
            if (rsi_1d_val < 30 and rsi_1w_val < 40 and rsi_4h_val < 35 and
                rsi_4h_val > rsi_4h[i-1] and bullish_rejection and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short setup: RSI divergence + overbought + bearish rejection
            elif (rsi_1d_val > 70 and rsi_1w_val > 60 and rsi_4h_val > 65 and
                  rsi_4h_val < rsi_4h[i-1] and bearish_rejection and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or breakdown
            if (rsi_4h_val > 70 or 
                close[i] < low[i-1] or  # Break below previous low
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or breakout
            if (rsi_4h_val < 30 or 
                close[i] > high[i-1] or  # Break above previous high
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1w_1d_RSI_Divergence_Volume_Rejection_v1"
timeframe = "4h"
leverage = 1.0