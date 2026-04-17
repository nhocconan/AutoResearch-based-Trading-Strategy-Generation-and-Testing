#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX(14) + 4h VWAP trend + volume spike confirmation.
# Uses 4h VWAP as trend filter (bullish when price > VWAP, bearish when price < VWAP),
# ADX > 25 to confirm trending market, and volume > 1.5x 20-period average for momentum.
# Designed to work in bull (trend up with volume) and bear (trend down with volume).
# Target: 20-30 trades/year to avoid fee drag on 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for VWAP calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h typical price and VWAP
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vwap_num = np.cumsum(typical_price_4h * volume_4h)
    vwap_den = np.cumsum(volume_4h)
    vwap_4h = vwap_num / vwap_den
    
    # Align 4h VWAP to 1h
    vwap_1h = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Calculate ADX(14) on 1h
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: spike > 1.5x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 35  # Need ADX(14) + VWAP + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1h[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        # Price relative to 4h VWAP
        price_above_vwap = close[i] > vwap_1h[i]
        price_below_vwap = close[i] < vwap_1h[i]
        
        if position == 0:
            # Long: Price above 4h VWAP with strong trend and volume
            if (price_above_vwap and strong_trend and volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price below 4h VWAP with strong trend and volume
            elif (price_below_vwap and strong_trend and volume_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 4h VWAP OR ADX drops below 20 (weakening trend)
            if (close[i] < vwap_1h[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses above 4h VWAP OR ADX drops below 20
            if (close[i] > vwap_1h[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_ADX14_4hVWAP_Volume"
timeframe = "1h"
leverage = 1.0