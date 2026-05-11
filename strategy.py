#!/usr/bin/env python3
"""
6h_OrderBook_Imbalance_12hTrend_Confirmation_v1
Hypothesis: Use 12h order book imbalance (buy/sell volume ratio) as a proxy for institutional bias, combined with 12h EMA50 trend filter and volume spike confirmation on 6h timeframe. In uptrends, buy when buying pressure exceeds selling; in downtrends, sell when selling pressure exceeds buying. Volume spike confirms conviction. Designed for low turnover (15-35 trades/year) to minimize fee drag while capturing sustained moves.
"""

name = "6h_OrderBook_Imbalance_12hTrend_Confirmation_v1"
timeframe = "6h"
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
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # === 12H Data for Trend and Order Book Imbalance ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    taker_buy_volume_12h = df_12h['taker_buy_volume'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h order book imbalance: buy volume / total volume (range 0-1)
    # >0.55 = buying pressure, <0.45 = selling pressure
    obi_12h = np.divide(
        taker_buy_volume_12h,
        volume_12h,
        out=np.full_like(taker_buy_volume_12h, 0.5),
        where=volume_12h!=0
    )
    
    # Align 12H indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    obi_12h_aligned = align_htf_to_ltf(prices, df_12h, obi_12h)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(obi_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: buying pressure > 55% AND uptrend (close > EMA50) AND volume spike
            if obi_12h_aligned[i] > 0.55 and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: selling pressure > 55% (obi < 0.45) AND downtrend (close < EMA50) AND volume spike
            elif obi_12h_aligned[i] < 0.45 and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks (close < EMA50) OR imbalance turns negative (obi < 0.5)
            if close[i] < ema_50_12h_aligned[i] or obi_12h_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: trend breaks (close > EMA50) OR imbalance turns positive (obi > 0.5)
            if close[i] > ema_50_12h_aligned[i] or obi_12h_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals