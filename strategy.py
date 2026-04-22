#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d RSI filter and volume confirmation.
# Long when price breaks above upper band + volume spike + 1d RSI > 50
# Short when price breaks below lower band + volume spike + 1d RSI < 50
# Bollinger squeeze identified when bandwidth < 20th percentile of last 50 periods.
# Works in trending markets (breakouts) and avoids false signals in choppy markets via squeeze filter.
# Target: 20-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align 1d RSI to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 4h Bollinger Bands (20, 2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_band - lower_band) / sma_20
    # Squeeze when bandwidth < 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_band[i]
        lower = lower_band[i]
        rsi = rsi_1d_aligned[i]
        is_squeeze = squeeze[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + volume squeeze + volume spike + RSI > 50
            if price > upper and is_squeeze and vol_spike and rsi > 50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + volume squeeze + volume spike + RSI < 50
            elif price < lower and is_squeeze and vol_spike and rsi < 50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to middle band (SMA20) or volatility expands
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to SMA20 or volatility expands significantly
                if price < sma_20[i] or bb_width[i] > bb_width_percentile[i] * 1.5:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to SMA20 or volatility expands significantly
                if price > sma_20[i] or bb_width[i] > bb_width_percentile[i] * 1.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Bollinger_Squeeze_Breakout_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0