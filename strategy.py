#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivots provide intraday support/resistance levels; breakout above H3 or below L3 with volume signals momentum
# Uses 4h EMA50 for trend filter (only trade in direction of higher timeframe momentum) to avoid counter-trend whipsaws
# Volume confirmation (>1.8x 20-period average) filters false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Designed for 15-37 trades/year on 1h timeframe to minimize fee drag while capturing intraday momentum
# Works in both bull and bear markets via 4h EMA50 trend filter - only trades with higher timeframe momentum

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend filter (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivots using prior day's range
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # We need prior day's OHLC - resample to 1d then align back
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla H3 and L3 levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 1h timeframe (already delayed by 1 day due to shift)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Session filter: 08-20 UTC (avoid low liquidity hours)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # warmup for volume MA and Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_h3 = camarilla_h3_aligned[i]
        curr_l3 = camarilla_l3_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below L3 (failed breakout)
            if curr_close < entry_price - 1.5 * curr_atr or curr_close < curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above H3 (failed breakout)
            if curr_close > entry_price + 1.5 * curr_atr or curr_close > curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry when price > 4h EMA50 (bullish regime) AND breaks above H3 with volume confirmation
            if curr_close > curr_ema50_4h and curr_high > curr_h3 and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry when price < 4h EMA50 (bearish regime) AND breaks below L3 with volume confirmation
            elif curr_close < curr_ema50_4h and curr_low < curr_l3 and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals