#!/usr/bin/env python3
name = "6h_Liquidity_Imbalance_Detection_12hTrendFilter"
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
    
    # Load 12H data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA20 on 12H for trend filter
    close_12h_s = pd.Series(close_12h)
    ema20_12h = close_12h_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate 6H EMA50 for liquidity imbalance detection
    close_s = pd.Series(close)
    ema50_6h = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6H RSI14 for overbought/oversold filter
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # First values
        if len(gain) >= period:
            avg_gain[period-1] = np.nanmean(gain[1:period])
            avg_loss[period-1] = np.nanmean(loss[1:period])
            
            # Wilder's smoothing
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.full_like(close, np.nan)
        valid = ~np.isnan(avg_loss) & (avg_loss != 0)
        rs[valid] = avg_gain[valid] / avg_loss[valid]
        
        rsi = np.full_like(close, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_6h = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(ema50_6h[i]) or 
            np.isnan(rsi_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12H EMA20
        trend_up = close[i] > ema20_12h_aligned[i]
        trend_down = close[i] < ema20_12h_aligned[i]
        
        # Liquidity imbalance: price deviation from 6H EMA50
        price_dev = (close[i] - ema50_6h[i]) / ema50_6h[i]
        # Long imbalance: price significantly below EMA50 (oversold)
        liq_long = price_dev < -0.008  # -0.8% deviation
        # Short imbalance: price significantly above EMA50 (overbought)
        liq_short = price_dev > 0.008   # +0.8% deviation
        
        # RSI filter: avoid extreme levels
        rsi_not_extreme = (rsi_6h[i] > 20) and (rsi_6h[i] < 80)
        
        if position == 0:
            # LONG: Uptrend + liquidity imbalance (oversold) + RSI not extreme
            if trend_up and liq_long and rsi_not_extreme:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + liquidity imbalance (overbought) + RSI not extreme
            elif trend_down and liq_short and rsi_not_extreme:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reverses or RSI overbought
            if (not trend_up) or (rsi_6h[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reverses or RSI oversold
            if (not trend_down) or (rsi_6h[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals