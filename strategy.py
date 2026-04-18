#!/usr/bin/env python3
"""
4h_CCI_RSI_Confluence
Hypothesis: Use CCI(20) and RSI(14) confluence on 4h timeframe with volume confirmation and ADX trend filter. CCI captures cyclical extremes while RSI measures momentum strength. In trending markets (ADX>25), we take signals in the direction of trend when both indicators show oversold/overbought conditions with volume confirmation. This reduces whipsaws in sideways markets and captures momentum in trending periods. Designed for 20-30 trades/year by requiring multiple confirmations.
"""

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
    
    # CCI calculation (20-period)
    typical_price = (high + low + close) / 3.0
    sma_tp = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    for i in range(20, n):
        sma_tp[i] = np.mean(typical_price[i-20:i+1])
        mad[i] = np.mean(np.abs(typical_price[i-20:i+1] - sma_tp[i]))
    
    cci = np.full(n, np.nan)
    for i in range(20, n):
        if mad[i] > 0:
            cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mad[i])
        else:
            cci[i] = 0.0
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[i-13:i+1])
            avg_loss[i] = np.mean(loss[i-13:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi = np.full(n, np.nan)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter (14-period) - using daily data for stability
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    tr = np.full(len(close_1d), np.nan)
    dm_plus = np.full(len(close_1d), np.nan)
    dm_minus = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
        
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0
            
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0
    
    # Smooth TR, DM+ and DM- (14-period)
    atr = np.full(len(close_1d), np.nan)
    dm_plus_smooth = np.full(len(close_1d), np.nan)
    dm_minus_smooth = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            atr[i] = np.mean(tr[1:i+1])
            dm_plus_smooth[i] = np.mean(dm_plus[1:i+1])
            dm_minus_smooth[i] = np.mean(dm_minus[1:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Calculate DI+ and DI-
    di_plus = np.full(len(close_1d), np.nan)
    di_minus = np.full(len(close_1d), np.nan)
    dx = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if atr[i] > 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if di_plus[i] + di_minus[i] > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
            else:
                dx[i] = 0
        else:
            di_plus[i] = 0
            di_minus[i] = 0
            dx[i] = 0
    
    # ADX is smoothed DX (14-period)
    adx = np.full(len(close_1d), np.nan)
    for i in range(28, len(close_1d)):
        if i == 28:
            adx[i] = np.mean(dx[15:i+1])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need CCI and RSI warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX indicates trending market (>25)
        if adx_aligned[i] > 25:
            if position == 0:
                # Long entry: CCI oversold (< -100) AND RSI oversold (< 30) with volume confirmation
                if (cci[i] < -100 and rsi[i] < 30 and vol_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short entry: CCI overbought (> 100) AND RSI overbought (> 70) with volume confirmation
                elif (cci[i] > 100 and rsi[i] > 70 and vol_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            
            elif position == 1:
                # Long exit: CCI becomes overbought OR RSI becomes overbought
                if (cci[i] > 100 or rsi[i] > 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Short exit: CCI becomes oversold OR RSI becomes oversold
                if (cci[i] < -100 or rsi[i] < 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging markets (ADX <= 25), stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "4h_CCI_RSI_Confluence"
timeframe = "4h"
leverage = 1.0