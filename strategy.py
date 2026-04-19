#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with weekly trend filter and volume confirmation.
# Long when: price closes above upper BB(20,2), close > weekly EMA20, volume > 1.5x 20-day average
# Short when: price closes below lower BB(20,2), close < weekly EMA20, volume > 1.5x 20-day average
# Exit when price returns to middle BB (20-period SMA)
# Bollinger Bands capture volatility expansion, weekly EMA filters trend, volume confirms breakout strength.
# Target: 15-25 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "1d_BollingerBreakout_WeeklyEMA_Volume"
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
    
    # Weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly data
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Bollinger Bands (20,2) on daily data
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    middle_band = sma20
    
    # 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations (20 for BB + 20 for weekly EMA alignment)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sma = sma20[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        ema20w = ema20_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price closes above upper BB, price > weekly EMA20, volume spike
            if (price > upper and close[i-1] <= upper_band[i-1] and  # close above upper band
                price > ema20w and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price closes below lower BB, price < weekly EMA20, volume spike
            elif (price < lower and close[i-1] >= lower_band[i-1] and  # close below lower band
                  price < ema20w and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle BB (mean reversion)
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle BB (mean reversion)
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals