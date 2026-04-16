#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Candlestick Pattern + 12h Trend Filter + Volume Spike
# Long when bullish engulfing pattern forms AND price > 12h EMA50 AND volume > 3x 24-period average volume
# Short when bearish engulfing pattern forms AND price < 12h EMA50 AND volume > 3x 24-period average volume
# Engulfing patterns provide high-probability reversal signals with clear entry/exit
# 12h EMA50 filter ensures alignment with intermediate trend, reducing counter-trend trades
# Volume spike adds conviction to pattern formations
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h EMA50 trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === Bullish Engulfing Pattern ===
    bullish_engulf = (close > open_price) & (open_price > np.roll(close, 1)) & (close > np.roll(open_price, 1))
    # === Bearish Engulfing Pattern ===
    bearish_engulf = (close < open_price) & (open_price < np.roll(close, 1)) & (close < np.roll(open_price, 1))
    
    # === Volume Spike (3x 24-period average) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > vol_ma_24 * 3.0
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        is_bullish_engulf = bullish_engulf[i]
        is_bearish_engulf = bearish_engulf[i]
        is_vol_spike = vol_spike[i]
        
        # === ENTRY LOGIC ===
        # Long when: bullish engulfing AND price > EMA50 AND volume spike
        if is_bullish_engulf and price > ema_val and is_vol_spike:
            signals[i] = 0.25
        # Short when: bearish engulfing AND price < EMA50 AND volume spike
        elif is_bearish_engulf and price < ema_val and is_vol_spike:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Engulfing_12hEMA50_Volume3x"
timeframe = "4h"
leverage = 1.0