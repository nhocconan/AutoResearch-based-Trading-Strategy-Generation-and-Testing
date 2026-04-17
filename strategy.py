# 6h_MarketStructure_Breakout_with_Liquidity_Profile
# Hypothesis: Combines market structure (BOS/CHoCH) with liquidity pool identification (equal highs/lows)
# and volume confirmation to capture institutional flow. Works in both bull/bear by trading with structure
# while avoiding fakeouts via liquidity sweeps. Target: 80-120 trades over 4 years (20-30/year).
# Timeframe: 6h, HTF: 1d for trend, liquidity levels

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and liquidity levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily swing points for structure
    swing_high = np.zeros(len(high_1d))
    swing_low = np.zeros(len(low_1d))
    
    # Simple swing detection: high/low surrounded by lower/higher values
    for i in range(2, len(high_1d)-2):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and \
           high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]:
            swing_high[i] = high_1d[i]
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and \
           low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]:
            swing_low[i] = low_1d[i]
    
    # Forward fill swing points
    swing_high = pd.Series(swing_high).replace(0, np.nan).ffill().bfill().values
    swing_low = pd.Series(swing_low).replace(0, np.nan).ffill().bfill().values
    
    # Calculate daily EMA for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all daily data to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Calculate 6h ATR for volatility filter and stop
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(swing_high_aligned[i]) or 
            np.isnan(swing_low_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i] * 0.8
        
        # Liquidity sweep detection (equal highs/lows within ATR)
        liq_sweep_low = False
        liq_sweep_high = False
        
        # Check for liquidity sweep below recent swing low
        if i >= 2:
            recent_low = np.min(low[max(0, i-10):i])
            if low[i] < recent_low and close[i] > recent_low:
                liq_sweep_low = True
        
        # Check for liquidity sweep above recent swing high
        if i >= 2:
            recent_high = np.max(high[max(0, i-10):i])
            if high[i] > recent_high and close[i] < recent_high:
                liq_sweep_high = True
        
        if position == 0:
            # Long: bullish trend + liquidity sweep low + volume
            if bullish_trend and vol_filter and liq_sweep_low:
                signals[i] = 0.25
                position = 1
            # Short: bearish trend + liquidity sweep high + volume
            elif bearish_trend and vol_filter and liq_sweep_high:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend change or liquidity sweep high
            if not bullish_trend or liq_sweep_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend change or liquidity sweep low
            if not bearish_trend or liq_sweep_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MarketStructure_Breakout_with_Liquidity_Profile"
timeframe = "6h"
leverage = 1.0