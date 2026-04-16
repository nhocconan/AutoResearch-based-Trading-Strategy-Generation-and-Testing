#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA200 filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets (2025+),
# buying near oversold (-80) and selling near overbought (-20) works well.
# Filtered by 1d EMA200 for trend alignment and volume > 1.3x average for confirmation.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-120 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === Williams %R on 4h (14-period) ===
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_4h) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # === 1d EMA200 for trend filter ===
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h volume confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_4h > (1.3 * vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema200 = ema200_1d_aligned[i]
        vol_conf = vol_confirm[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Williams %R reaches overbought (-20) or Alligator death cross (not used here)
            if wr >= -20.0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R reaches oversold (-80)
            if wr <= -80.0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol_conf:
                # Go long when oversold (-80) and price above 1d EMA200 (uptrend bias)
                if wr <= -80.0 and price > ema200:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when overbought (-20) and price below 1d EMA200 (downtrend bias)
                elif wr >= -20.0 and price < ema200:
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

name = "4h_WilliamsR_1dEMA200_Volume_Filter"
timeframe = "4h"
leverage = 1.0