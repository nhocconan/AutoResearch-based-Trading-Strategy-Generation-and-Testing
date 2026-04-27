#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicator calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Calculate daily RSI(14) for overbought/oversold filter
    rsi_14_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        avg_gain[14] = np.mean(gain[:14])
        avg_loss[14] = np.mean(loss[:14])
        for i in range(15, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi_14_1d[i] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands(20,2) for volatility regime
    bb_upper_1d = np.full(len(close_1d), np.nan)
    bb_lower_1d = np.full(len(close_1d), np.nan)
    bb_middle_1d = np.full(len(close_1d), np.nan)
    bb_width_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            bb_middle_1d[i] = np.mean(close_1d[i-19:i+1])
            bb_std = np.std(close_1d[i-19:i+1])
            bb_upper_1d[i] = bb_middle_1d[i] + 2 * bb_std
            bb_lower_1d[i] = bb_middle_1d[i] - 2 * bb_std
            bb_width_1d[i] = (bb_upper_1d[i] - bb_lower_1d[i]) / bb_middle_1d[i] if bb_middle_1d[i] != 0 else 0
    
    # Align daily indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Calculate 4h ATR(14) for volatility filter and stoploss
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(34, vol_period, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(bb_width_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        # Bollinger width regime filter: narrow bands = low volatility (good for breakouts)
        vol_regime_filter = bb_width_1d_aligned[i] < 0.05
        
        if position == 0:
            # Long: Price breaks above upper BB with volume, RSI not overbought, and above daily EMA34
            if (price > bb_upper_1d_aligned[i] and vol_filter and 
                rsi_14_1d_aligned[i] < 70 and price > ema_34_1d_aligned[i] and vol_regime_filter):
                signals[i] = size
                position = 1
            # Short: Price breaks below lower BB with volume, RSI not oversold, and below daily EMA34
            elif (price < bb_lower_1d_aligned[i] and vol_filter and 
                  rsi_14_1d_aligned[i] > 30 and price < ema_34_1d_aligned[i] and vol_regime_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below middle BB or trailing stop
            if price < bb_middle_1d_aligned[i] or price < ema_34_1d_aligned[i] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above middle BB or trailing stop
            if price > bb_middle_1d_aligned[i] or price > ema_34_1d_aligned[i] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger_Breakout_EMA34_RSI_Volume"
timeframe = "4h"
leverage = 1.0