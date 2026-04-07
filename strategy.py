#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4H Donchian channel breakout with volume confirmation and daily trend filter
# Uses Donchian(20) for breakout signals, volume ratio for confirmation, and daily EMA(50) for trend filter.
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
# Works in bull markets via breakout momentum and in bear via mean reversion at channel extremes.

name = "4h_donchian20_volume_ema50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current / 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Breakout conditions with volume confirmation
        upper_breakout = close[i] > donch_high[i] and vol_ratio[i] > 1.5
        lower_breakout = close[i] < donch_low[i] and vol_ratio[i] > 1.5
        
        # Mean reversion at channel extremes (for ranging markets)
        channel_width = donch_high[i] - donch_low[i]
        if channel_width > 0:
            position_in_channel = (close[i] - donch_low[i]) / channel_width
            oversold = position_in_channel < 0.2 and vol_ratio[i] > 1.2
            overbought = position_in_channel > 0.8 and vol_ratio[i] > 1.2
        else:
            oversold = False
            overbought = False
        
        # Position sizing
        base_size = 0.25
        
        # Long conditions: bullish breakout OR oversold mean reversion in uptrend
        if (upper_breakout and bullish_trend) or (oversold and bullish_trend):
            signals[i] = base_size
        # Short conditions: bearish breakout OR overbought mean reversion in downtrend
        elif (lower_breakout and bearish_trend) or (overbought and bearish_trend):
            signals[i] = -base_size
        else:
            signals[i] = 0.0
    
    return signals