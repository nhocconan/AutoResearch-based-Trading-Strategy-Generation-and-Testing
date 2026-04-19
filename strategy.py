#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Keltner breakout with 4h trend filter and volume confirmation.
# Long when price breaks above upper Keltner band (EMA20 + 2*ATR10) with price above 4h EMA50 and volume spike (>1.5x average).
# Short when price breaks below lower Keltner band (EMA20 - 2*ATR10) with price below 4h EMA50 and volume spike.
# Uses 4h EMA50 as trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 15-37 trades/year per symbol (~60-150 total over 4 years).
name = "1h_Keltner_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h close
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (wait for 4h close)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Keltner bands on 1h data
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = pd.Series(np.maximum.reduce([
        high - low,
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ])).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10, 50)  # Need EMA20, ATR10, and 4h EMA50 data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_50_aligned[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper Keltner AND above 4h EMA50
            if price > upper and price > ema_trend and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below lower Keltner AND below 4h EMA50
            elif price < lower and price < ema_trend and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower Keltner or below 4h EMA50
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price breaks above upper Keltner or above 4h EMA50
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals