#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR(14) with proper handling of first value
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR with Wilder's smoothing (alpha = 1/14)
    atr_1d = np.full(len(tr), np.nan)
    atr_1d[13] = np.mean(tr[0:14])  # Seed with first 14 values
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = np.full(len(close_1d), np.nan)
    ema_34_1d[33] = np.mean(close_1d[0:34])  # Seed
    for i in range(34, len(close_1d)):
        ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Align daily EMA to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4-period RSI for momentum confirmation
    def calculate_rsi(close_prices, period=4):
        rsi = np.full(len(close_prices), np.nan)
        if len(close_prices) < period + 1:
            return rsi
        
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close_prices), np.nan)
        avg_loss = np.full(len(close_prices), np.nan)
        
        avg_gain[period] = np.mean(gain[0:period])
        avg_loss[period] = np.mean(loss[0:period])
        
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
            
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    rsi_4 = calculate_rsi(close, 4)
    
    # Calculate 4h ATR for position sizing and stop loss
    tr_4h_1 = high - low
    tr_4h_2 = np.abs(high - np.roll(close, 1))
    tr_4h_3 = np.abs(low - np.roll(close, 1))
    tr_4h_1[0] = high[0] - low[0]
    tr_4h_2[0] = np.abs(high[0] - close[0])
    tr_4h_3[0] = np.abs(low[0] - close[0])
    tr_4h = np.maximum(tr_4h_1, np.maximum(tr_4h_2, tr_4h_3))
    
    atr_4h = np.full(len(tr_4h), np.nan)
    atr_4h[13] = np.mean(tr_4h[0:14])
    for i in range(14, len(tr_4h)):
        atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi_4[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price above daily EMA34 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Momentum filter: RSI not extreme
        momentum_ok = (rsi_4[i] > 20) and (rsi_4[i] < 80)
        
        if position == 0:
            # Long entry: price above open + 0.25*ATR, with volume, trend and momentum filters
            if (close[i] > open_price[i] + 0.25 * atr_4h[i] and 
                vol_confirmed and 
                trend_up and 
                momentum_ok):
                signals[i] = 0.25
                position = 1
            # Short entry: price below open - 0.25*ATR, with volume, trend and momentum filters
            elif (close[i] < open_price[i] - 0.25 * atr_4h[i] and 
                  vol_confirmed and 
                  trend_down and 
                  momentum_ok):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below open or ATR-based stop
            if close[i] < open_price[i] - 1.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above open or ATR-based stop
            if close[i] > open_price[i] + 1.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA34Daily_RSI4_VolumeFilter"
timeframe = "4h"
leverage = 1.0