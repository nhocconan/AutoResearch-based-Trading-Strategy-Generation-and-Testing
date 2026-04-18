#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_prices = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 50-period EMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get daily data for 20-period ATR (volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR20 on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Get weekly data for 14-period RSI (overbought/oversold filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1w = np.divide(avg_gain_1w, avg_loss_1w, out=np.zeros_like(avg_gain_1w), where=avg_loss_1w!=0)
    rsi_14_1w = 100 - (100 / (1 + rs_1w))
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate 6-period RSI on 6h timeframe (momentum filter)
    delta_6h = np.diff(close, prepend=close[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    avg_gain_6h = pd.Series(gain_6h).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss_6h = pd.Series(loss_6h).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs_6h = np.divide(avg_gain_6h, avg_loss_6h, out=np.zeros_like(avg_gain_6h), where=avg_loss_6h!=0)
    rsi_6_6h = 100 - (100 / (1 + rs_6h))
    
    # Calculate 6h ATR for stop loss
    tr_6h_1 = high - low
    tr_6h_2 = np.abs(high - np.roll(close, 1))
    tr_6h_3 = np.abs(low - np.roll(close, 1))
    tr_6h_1[0] = high[0] - low[0]
    tr_6h_2[0] = np.abs(high[0] - close[0])
    tr_6h_3[0] = np.abs(low[0] - close[0])
    tr_6h = np.maximum(tr_6h_1, np.maximum(tr_6h_2, tr_6h_3))
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14, 6)  # need EMA50, ATR20, RSI14w, RSI6h
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_20_1d_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i]) or np.isnan(rsi_6_6h[i]) or 
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: current 6h ATR < 1.5 * daily ATR20 (avoid high volatility)
        vol_filter = atr_6h[i] < 1.5 * atr_20_1d_aligned[i]
        
        # Momentum filter: RSI6 not in extreme territory
        momentum_filter = (rsi_6_6h[i] > 20) and (rsi_6_6h[i] < 80)
        
        # Overbought/oversold filter: weekly RSI not extreme
        rsi_filter = (rsi_14_1w_aligned[i] > 30) and (rsi_14_1w_aligned[i] < 70)
        
        if position == 0:
            # Long entry: price above EMA50, with filters
            if (price_above_ema and vol_filter and momentum_filter and rsi_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price below EMA50, with filters
            elif (price_below_ema and vol_filter and momentum_filter and rsi_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below EMA50 or volatility spike
            if (close[i] < ema_50_1d_aligned[i]) or (atr_6h[i] > 2.0 * atr_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA50 or volatility spike
            if (close[i] > ema_50_1d_aligned[i]) or (atr_6h[i] > 2.0 * atr_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA50Daily_ATR20Daily_RSIFilter"
timeframe = "6h"
leverage = 1.0