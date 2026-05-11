#!/usr/bin/env python3
name = "4h_1W_Financial_Strength_Index_Reversion"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1W data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Financial Strength Index (FSI) on weekly data
    # FSI = (RSI(14) + MFI(14)) / 2
    # RSI calculation
    delta = np.diff(df_1w['close'], prepend=df_1w['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # MFI calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    raw_money = typical_price * df_1w['volume']
    money_flow = np.where(typical_price > np.roll(typical_price, 1), raw_money, 
                         np.where(typical_price < np.roll(typical_price, 1), -raw_money, 0))
    
    pos_mf = np.where(money_flow > 0, money_flow, 0)
    neg_mf = np.where(money_flow < 0, -money_flow, 0)
    
    # Wilder's smoothing for MFI
    pos_mf_14 = np.zeros_like(pos_mf)
    neg_mf_14 = np.zeros_like(neg_mf)
    pos_mf_14[13] = np.sum(pos_mf[1:14])
    neg_mf_14[13] = np.sum(neg_mf[1:14])
    for i in range(14, len(pos_mf)):
        pos_mf_14[i] = pos_mf_14[i-1] + pos_mf[i] - pos_mf[i-13]
        neg_mf_14[i] = neg_mf_14[i-1] + neg_mf[i] - neg_mf[i-13]
    
    mfi = np.where((pos_mf_14 + neg_mf_14) != 0, 100 * pos_mf_14 / (pos_mf_14 + neg_mf_14), 50)
    
    # FSI = average of RSI and MFI
    fsi = (rsi + mfi) / 2
    
    # Align FSI to 4h timeframe with extra delay for confirmation
    fsi_aligned = align_htf_to_ltf(prices, df_1w, fsi, additional_delay_bars=1)
    
    # 4H EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(fsi_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: FSI oversold (<30) with volume surge and price above EMA20
            if (fsi_aligned[i] < 30 and 
                volume_surge and 
                close[i] > ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: FSI overbought (>70) with volume surge and price below EMA20
            elif (fsi_aligned[i] > 70 and 
                  volume_surge and 
                  close[i] < ema_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: FSI returns to neutral (>50) or trend turns bearish
                if (fsi_aligned[i] > 50) or (close[i] < ema_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: FSI returns to neutral (<50) or trend turns bullish
                if (fsi_aligned[i] < 50) or (close[i] > ema_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals