#!/usr/bin/env python3
# 12h_1d_Hybrid_Strategy: Combines 1d trend (EMA34), momentum (RSI14), and volume confirmation with 12h price action
# Uses 12h timeframe for lower trade frequency and better risk-adjusted returns
# Designed to work in both bull and bear markets via trend/momentum alignment and volume filters
# Target: 15-30 trades/year to minimize fee drag on 12h timeframe

name = "12h_1d_Hybrid_Strategy"
timeframe = "12h"
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
    
    # 1d trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 1d momentum filter (RSI14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi14_1d = 100 - (100 / (1 + rs))
    rsi_overbought = rsi14_1d > 70
    rsi_oversold = rsi14_1d < 30
    
    # Align 1d RSI to 12h
    rsi_overbought_aligned = align_htf_to_ltf(prices, df_1d, rsi_overbought.astype(float))
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_1d, rsi_oversold.astype(float))
    
    # Volume confirmation (1.5x 24-period average on 12h)
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma[i] = vol_sum / 24
        else:
            vol_ma[i] = np.nan
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(rsi_overbought_aligned[i]) or np.isnan(rsi_oversold_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA34, RSI not overbought, volume confirmation
            if (close[i] > ema34_1d[-1] if len(ema34_1d) > 0 else False and  # Simplified: use current 1d EMA
                not rsi_overbought_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA34, RSI not oversold, volume confirmation
            elif (close[i] < ema34_1d[-1] if len(ema34_1d) > 0 else False and  # Simplified: use current 1d EMA
                  not rsi_oversold_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below EMA34 or RSI overbought
            if (close[i] < ema34_1d[-1] if len(ema34_1d) > 0 else False or
                rsi_overbought_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above EMA34 or RSI oversold
            if (close[i] > ema34_1d[-1] if len(ema34_1d) > 0 else False or
                rsi_oversold_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals