#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Bollinger Band Width with 1-week EMA trend filter and volume confirmation.
# Bollinger Band Width (BBW) = (Upper Band - Lower Band) / Middle Band (SMA20)
# Long when: BBW > 0.05 (expansion) and price > upper band and weekly EMA50 rising and volume > 1.5x average
# Short when: BBW > 0.05 (expansion) and price < lower band and weekly EMA50 falling and volume > 1.5x average
# Exit when: price crosses back inside Bollinger Bands (middle band)
# BBW expansion signals increasing volatility and potential trend continuation.
# Weekly EMA50 filter ensures alignment with higher timeframe trend.
# Volume confirmation ensures institutional participation.
# Works in bull (buy strength) and bear (sell weakness) by capturing volatility expansion breakouts.
# Target: 15-25 trades/year per symbol.
name = "1d_BollingerWidth_Expansion_Volume"
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
    
    # Calculate Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    bb_width = (upper_band - lower_band) / sma20  # Normalized width
    
    # Get weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for BB calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(bb_width[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        bbw = bb_width[i]
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        sma = sma20[i]
        ema50 = ema50_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Weekly EMA trend direction (rising/falling)
        ema50_rising = ema50 > ema50_1w_aligned[i-1] if i > 0 else False
        ema50_falling = ema50 < ema50_1w_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long entry: BBW expansion + price > upper band + weekly EMA50 rising + volume spike
            if (bbw > 0.05 and price > upper and ema50_rising and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: BBW expansion + price < lower band + weekly EMA50 falling + volume spike
            elif (bbw > 0.05 and price < lower and ema50_falling and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back inside Bollinger Bands (below middle band)
            if price < sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back inside Bollinger Bands (above middle band)
            if price > sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals