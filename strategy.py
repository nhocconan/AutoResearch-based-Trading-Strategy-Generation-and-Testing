#!/usr/bin/env python3
"""
6h_parabolic_sar_trend_v1
Hypothesis: Parabolic SAR on 6h timeframe with 1d trend filter (price above/below 200 EMA) captures trends effectively in both bull and bear markets. Uses acceleration factor 0.02, max 0.2. Only takes longs when price > 200 EMA and SAR flips below price; shorts when price < 200 EMA and SAR flips above price. Volume confirmation filters weak signals. Designed for 15-35 trades/year to minimize fee dust while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_parabolic_sar_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parabolic SAR calculation
    def calculate_psar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
        n = len(close)
        psar = np.full(n, np.nan)
        bull = True  # True for long, False for short
        af = af_start
        ep = low[0] if bull else high[0]  # extreme point
        psar[0] = ep
        
        for i in range(1, n):
            if bull:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                # Reverse if price touches SAR
                if low[i] <= psar[i]:
                    bull = False
                    psar[i] = ep  # SAR becomes prior EP
                    af = af_start
                    ep = high[i]
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
            else:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                # Reverse if price touches SAR
                if high[i] >= psar[i]:
                    bull = True
                    psar[i] = ep  # SAR becomes prior EP
                    af = af_start
                    ep = low[i]
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)
        return psar
    
    psar = calculate_psar(high, low, close)
    
    # 200 EMA for trend filter
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d trend filter: price vs 200 EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not available
        if (np.isnan(psar[i]) or np.isnan(ema_200[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: SAR flips above price (trend reversal)
            if psar[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: SAR flips below price (trend reversal)
            if psar[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: SAR flips below price AND price > 200 EMA (1d)
                if psar[i] < close[i] and psar[i-1] >= close[i-1] and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: SAR flips above price AND price < 200 EMA (1d)
                elif psar[i] > close[i] and psar[i-1] <= close[i-1] and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals