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
    
    # === Daily data (HTF) for regime detection ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate daily ATR(14) for volatility regime
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'],
                       np.maximum(np.abs(df_1d['high'] - np.roll(close_1d, 1)),
                                  np.abs(df_1d['low'] - np.roll(close_1d, 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ATR(14) on 4h for position sizing and stoploss
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate Bollinger Bands width on 4h for regime detection
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20_4h + (2 * std_20_4h)
    bb_lower = sma_20_4h - (2 * std_20_4h)
    bb_width = (bb_upper - bb_lower) / sma_20_4h
    bb_width_aligned = bb_width  # Already on 4h timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or np.isnan(bb_width_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        price = close_4h[i]
        ema_200_val = ema_200_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        bb_width_val = bb_width_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below daily EMA(200) OR trailing stop hit
            if (price < ema_200_val) or (price < entry_price - 2.0 * atr_4h_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above daily EMA(200) OR trailing stop hit
            if (price > ema_200_val) or (price > entry_price + 2.0 * atr_4h_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine volatility regime: low volatility when BB width < 20th percentile
            # Use historical BB width to calculate percentile
            if i >= 50:  # Need sufficient history for percentile
                bb_width_history = bb_width_aligned[max(0, i-50):i]
                bb_width_percentile = (bb_width_history < bb_width_val).sum() / len(bb_width_history) * 100
                low_volatility = bb_width_percentile < 20
            else:
                low_volatility = False
            
            # Only trade in low volatility regimes (squeeze breakout)
            if low_volatility:
                # LONG: Price above daily EMA(200) AND breaking above recent high
                if (price > ema_200_val) and (price > np.max(high_4h[max(0, i-20):i])):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                
                # SHORT: Price below daily EMA(200) AND breaking below recent low
                elif (price < ema_200_val) and (price < np.min(low_4h[max(0, i-20):i])):
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyEMA200_SqueezeBreakout_ATRStop"
timeframe = "4h"
leverage = 1.0