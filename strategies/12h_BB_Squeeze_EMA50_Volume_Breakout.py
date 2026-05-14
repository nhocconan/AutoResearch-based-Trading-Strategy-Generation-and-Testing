#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d EMA50 trend filter and volume confirmation.
# Bollinger Band squeeze identifies low volatility periods preceding explosive moves.
# We trade breakouts in the direction of the daily trend (EMA50) with volume confirmation.
# Works in bull/bear markets: avoids false breakouts in ranging markets, captures true breakouts.
# Target: 15-35 trades/year per symbol.
name = "12h_BB_Squeeze_EMA50_Volume_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Bollinger Bands (20, 2) on 12h
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = (upper_band - lower_band) / sma_20  # Normalized bandwidth
    
    # Bollinger Band squeeze detection: bandwidth below 20-period mean
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma  # Bandwidth below average = squeeze
    
    # Align 1d EMA50 to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 20)  # Ensure BB, EMA50, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        is_squeeze = squeeze_condition[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper  # Price breaks above upper band
        bearish_breakout = price < lower  # Price breaks below lower band
        
        if position == 0:
            # Look for entry after Bollinger Band squeeze, in direction of daily trend
            if is_squeeze and bullish_breakout and (price > ema_50_val) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif is_squeeze and bearish_breakout and (price < ema_50_val) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to middle band (mean reversion) or volatility expands
            middle_band = sma_20[i]
            if price < middle_band:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle band
            middle_band = sma_20[i]
            if price > middle_band:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals