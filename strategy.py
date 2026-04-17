#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band (20,2.0) breakout with 1d RSI(14) filter and volume confirmation.
# Enters long when price breaks above upper band with volume and RSI > 50 (bullish momentum).
# Enters short when price breaks below lower band with volume and RSI < 50 (bearish momentum).
# Exits when price reverts to middle band (mean reversion) or RSI crosses 50.
# Designed for low turnover (target: 12-37 trades/year) using volatility-based breakouts.
# Works in bull markets (breakouts with momentum) and bear markets (mean reversion at bands).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 12h Bollinger Bands (20,2.0)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_12h = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Volume filter: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or 
            np.isnan(rsi_1d_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.8x average (strict to reduce trades)
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # RSI filter: >50 for long, <50 for short
        rsi_bullish = rsi_1d_12h[i] > 50
        rsi_bearish = rsi_1d_12h[i] < 50
        
        # Price relative to Bollinger Bands
        price_above_upper = close[i] > bb_upper[i]
        price_below_lower = close[i] < bb_lower[i]
        price_above_middle = close[i] > bb_middle[i]
        price_below_middle = close[i] < bb_middle[i]
        
        if position == 0:
            # Long: Price breaks above upper band with volume and bullish RSI
            if (price_above_upper and volume_filter and rsi_bullish):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with volume and bearish RSI
            elif (price_below_lower and volume_filter and rsi_bearish):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reverts to middle band OR RSI turns bearish
            if (price_below_middle) or (not rsi_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reverts to middle band OR RSI turns bullish
            if (price_above_middle) or (not rsi_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BollingerBreakout_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0