#!/usr/bin/env python3
# 4h_ThreeLineBreak_Trend_Follow
# Hypothesis: Three Line Break (TLB) on 12h defines trend, with 4h RSI for entry timing and ATR-based stop.
# TLB filters out noise and captures sustained moves. Works in bull/bear by following established trend.
# Targets ~25-35 trades/year to minimize fee drag.

name = "4h_ThreeLineBreak_Trend_Follow"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for Three Line Break trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 3:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Three Line Break calculation
    tlb_direction = np.zeros(len(close_12h), dtype=int)  # 1 for up, -1 for down, 0 for reverse
    tlb_blocks = []  # store closing prices of each block
    current_direction = 0
    
    for i in range(len(close_12h)):
        price = close_12h[i]
        
        if len(tlb_blocks) == 0:
            tlb_blocks.append(price)
            tlb_direction[i] = 1  # start with up
            current_direction = 1
        else:
            if current_direction == 1:  # in up trend
                if price > tlb_blocks[-1]:  # continue up
                    tlb_blocks.append(price)
                    tlb_direction[i] = 1
                elif price < min(tlb_blocks[-3:] if len(tlb_blocks) >= 3 else tlb_blocks):  # reverse down
                    tlb_blocks = [price]
                    tlb_direction[i] = -1
                    current_direction = -1
                else:  # no change
                    tlb_direction[i] = 0
            else:  # in down trend
                if price < tlb_blocks[-1]:  # continue down
                    tlb_blocks.append(price)
                    tlb_direction[i] = -1
                elif price > max(tlb_blocks[-3:] if len(tlb_blocks) >= 3 else tlb_blocks):  # reverse up
                    tlb_blocks = [price]
                    tlb_direction[i] = 1
                    current_direction = 1
                else:  # no change
                    tlb_direction[i] = 0
    
    # Align 12h TLB direction to 4h
    tlb_direction_float = tlb_direction.astype(float)
    tlb_up_aligned = align_htf_to_ltf(prices, df_12h, (tlb_direction_float == 1).astype(float))
    tlb_down_aligned = align_htf_to_ltf(prices, df_12h, (tlb_direction_float == -1).astype(float))
    
    # 4h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(tlb_up_aligned[i]) or np.isnan(tlb_down_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h TLB up + RSI not overbought + volatility filter
            if (tlb_up_aligned[i] > 0.5 and
                rsi[i] < 70 and
                atr[i] > 0):  # volatility present
                signals[i] = 0.25
                position = 1
            # Short: 12h TLB down + RSI not oversold + volatility filter
            elif (tlb_down_aligned[i] > 0.5 and
                  rsi[i] > 30 and
                  atr[i] > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: 12h TLB reverses down or RSI extremely overbought
            if (tlb_down_aligned[i] > 0.5 or
                rsi[i] > 85):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: 12h TLB reverses up or RSI extremely oversold
            if (tlb_up_aligned[i] > 0.5 or
                rsi[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals