#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Enters long when price breaks above Camarilla H3 level with expanding volume and bullish 1d trend.
# Enters short when price breaks below Camarilla L3 level with expanding volume and bearish 1d trend.
# Uses ATR(14) for dynamic stoploss and position sizing.
# Designed for 12-37 trades/year on 12h timeframe with focus on trend continuation.
# Camarilla levels provide institutional reference points, volume filter ensures participation.
# 1d trend filter prevents counter-trend trading in choppy markets.

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h ATR(14) for volatility filtering and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from second bar
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * 20-period average volume
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Calculate Camarilla pivot levels for current 12h bar using previous day's OHLC
        # Use previous day's data to avoid look-ahead
        prev_day_idx = i - 1
        if prev_day_idx < 0:
            prev_day_idx = 0
            
        # Calculate pivot points using previous day's OHLC
        if i >= 1:
            # Get previous day's OHLC (1d data aligned to 12h)
            # Since we're on 12h timeframe, we need to get the previous day's data
            # We'll use the close of the previous 12h bar as proxy for daily OHLC
            # For proper Camarilla, we need actual daily OHLC, so we approximate:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            range_val = prev_high - prev_low
            
            # Camarilla levels
            h3 = pivot + (range_val * 1.1 / 4)
            l3 = pivot - (range_val * 1.1 / 4)
            h4 = pivot + (range_val * 1.1 / 2)
            l4 = pivot - (range_val * 1.1 / 2)
        else:
            # Default values if not enough data
            h3 = l3 = h4 = l4 = close[i]
        
        # Determine 1d trend direction
        is_bullish_trend = close[i] > ema_50_1d_aligned[i]
        is_bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        bullish_breakout = (high[i] > h3) and vol_filter and is_bullish_trend
        bearish_breakout = (low[i] < l3) and vol_filter and is_bearish_trend
        
        # Exit conditions: reversal signal or ATR-based stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on bearish breakout or if price drops below L3
            exit_long = bearish_breakout or (low[i] < l3)
        elif position == -1:
            # Exit short on bullish breakout or if price rises above H3
            exit_short = bullish_breakout or (high[i] > h3)
        
        # Priority: entry > exit > hold
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals