#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12-hour volume-weighted average price (VWAP) with 12-hour EMA200 trend filter and volume confirmation.
In uptrend (price > 12h EMA200), buy when price crosses above 12h VWAP with volume spike; in downtrend (price < 12h EMA200), sell when price crosses below 12h VWAP with volume spike.
Volume must exceed 1.5x 20-period average to confirm breakout strength. Exit on trend reversal or 2x ATR stop.
Designed for 20-40 trades/year (80-160 total over 4 years) to balance opportunity and fee drag.
VWAP acts as dynamic support/resistance, and EMA200 filters for major trend direction.
Works in bull markets via VWAP bounces in uptrend and in bear markets via VWAP rejections in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for VWAP and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high_12h + low_12h + close_12h) / 3.0
    pv = typical_price * volume_12h
    cum_pv = np.nancumsum(pv)
    cum_volume = np.nancumsum(volume_12h)
    vwap = np.divide(cum_pv, cum_volume, out=np.full_like(cum_pv, np.nan), where=cum_volume!=0)
    
    # Calculate 12h EMA200 for trend filter
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h indicators to 4h timeframe (wait for 12h bar to close)
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        vwap_val = vwap_aligned[i]
        ema_trend = ema_200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price crosses above VWAP + uptrend + volume spike
            if (price_open <= vwap_val and price_close > vwap_val and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below VWAP + downtrend + volume spike
            elif (price_open >= vwap_val and price_close < vwap_val and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR ATR-based stoploss
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from VWAP at entry time)
            if position == 1:
                # Use VWAP at entry as reference (approximated by current VWAP for simplicity)
                if price_close < vwap_val - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                if price_close > vwap_val + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_EMA200_Volume_ATR"
timeframe = "4h"
leverage = 1.0