#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h EMA34 trend filter and volume confirmation.
# Long when: BB width < 20th percentile (squeeze), price breaks above upper band, price > EMA34(12h), volume > 1.5x average
# Short when: BB width < 20th percentile (squeeze), price breaks below lower band, price < EMA34(12h), volume > 1.5x average
# Exit when price returns to middle band (20-period SMA)
# Bollinger squeeze identifies low volatility breakouts, EMA34 filters trend direction, volume confirms breakout strength.
# Works in bull (buy breakouts above squeeze) and bear (sell breakdowns below squeeze).
name = "6h_BB_Squeeze_EMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h data
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Bollinger Bands (20, 2) on 6h data
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    bb_width = (upper_band - lower_band) / sma20  # Normalized width
    
    # 20th percentile of BB width for squeeze detection (using expanding window to avoid look-ahead)
    bb_width_pct = np.full_like(bb_width, np.nan)
    for i in range(20, len(bb_width)):
        bb_width_pct[i] = np.percentile(bb_width[:i+1], 20)
    
    # Volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_width[i]) or np.isnan(bb_width_pct[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bbw = bb_width[i]
        bbw_pct = bb_width_pct[i]
        ema34 = ema34_12h_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: BB squeeze breakout above upper band, price > EMA34, volume confirmation
            if (bbw < bbw_pct and  # Squeeze condition
                price > upper_band[i] and  # Break above upper band
                price > ema34 and  # Above trend filter
                vol > 1.5 * vol_ma):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short entry: BB squeeze breakout below lower band, price < EMA34, volume confirmation
            elif (bbw < bbw_pct and  # Squeeze condition
                  price < lower_band[i] and  # Break below lower band
                  price < ema34 and  # Below trend filter
                  vol > 1.5 * vol_ma):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band (20-period SMA)
            if price < sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band (20-period SMA)
            if price > sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals