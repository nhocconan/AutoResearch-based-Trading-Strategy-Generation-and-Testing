#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter
# Long when price breaks above Donchian(20) upper band AND price > 1d EMA50 AND ATR(14) < 0.03 * price (low volatility)
# Short when price breaks below Donchian(20) lower band AND price < 1d EMA50 AND ATR(14) < 0.03 * price
# Exit when price retouches Donchian(20) midpoint or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 30-60 trades/year on 4h.
# Donchian breakouts capture strong momentum moves. 1d EMA50 ensures we trade with the dominant trend.
# ATR volatility filter avoids false breakouts during high-chop periods, improving win rate in both bull and bear markets.

name = "4h_Donchian20_1dEMA50_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar: no previous close
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 50)  # Donchian(20), ATR(14), and EMA50 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_band = highest_20[i]
        lower_band = lowest_20[i]
        midpoint = donchian_mid[i]
        ema_50 = ema_50_1d_aligned[i]
        atr = atr_14[i]
        
        # Volatility filter: ATR < 3% of price (avoid high-chop environments)
        vol_filter = atr < 0.03 * curr_close
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND price > 1d EMA50 AND low volatility
            if curr_high > upper_band and curr_close > ema_50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower band AND price < 1d EMA50 AND low volatility
            elif curr_low < lower_band and curr_close < ema_50 and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches midpoint or breaks below lower band
            if curr_close <= midpoint or curr_low < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches midpoint or breaks above upper band
            if curr_close >= midpoint or curr_high > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals