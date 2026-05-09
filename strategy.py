#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout (20-period) with weekly trend filter and volume confirmation.
# Long when price breaks above upper band + volume spike + price above 1w EMA(50).
# Short when price breaks below lower band + volume spike + price below 1w EMA(50).
# Uses ATR-based stop loss to limit drawdown. Designed to capture trends while avoiding whipsaw.
# Focus on higher timeframe (1w) trend to work in both bull and bear markets.
name = "1d_DonchianBreakout_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA trend filter (50-period)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ema20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: break above upper band + volume + price above 1w EMA
            if price > highest_high[i] and vol_confirm[i] and price > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below lower band + volume + price below 1w EMA
            elif price < lowest_low[i] and vol_confirm[i] and price < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: exit on stop loss or reverse signal
            if price < entry_price - 2.5 * atr[i]:  # stop loss
                signals[i] = 0.0
                position = 0
            elif price < lowest_low[i]:  # reverse signal (break below lower band)
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit on stop loss or reverse signal
            if price > entry_price + 2.5 * atr[i]:  # stop loss
                signals[i] = 0.0
                position = 0
            elif price > highest_high[i]:  # reverse signal (break above upper band)
                signals[i] = 0.25
                position = 1
                entry_price = price
            else:
                signals[i] = -0.25
    
    return signals