#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
# - Uses 1d Camarilla pivot levels (R3, R4, S3, S4) for breakout/fade logic
# - Long: Price breaks above R4 with volume > 1.5x 20-period 1d average volume
# - Short: Price breaks below S4 with volume > 1.5x 20-period 1d average volume
# - Fade trades: Long at S3 with volume confirmation, Short at R3 with volume confirmation
# - Exit: Opposite Camarilla level (R3 for long exits, S3 for short exits) or ATR trailing stop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in both ranging and trending markets
# - Volume confirmation filters out weak breakouts and increases signal quality
# - 6h timeframe balances trade frequency and responsiveness for BTC/ETH

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + range_1d * 1.1 / 4.0
    r4 = pp + range_1d * 1.1 / 2.0
    s3 = pp - range_1d * 1.1 / 4.0
    s4 = pp - range_1d * 1.1 / 2.0
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute ATR for stoploss (6h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        r3_level = r3_aligned[i]
        r4_level = r4_aligned[i]
        s3_level = s3_aligned[i]
        s4_level = s4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Breakout entries
        # Long breakout: price closes above R4 with volume confirmation
        if close_price > r4_level and vol_confirm:
            enter_long = True
        # Short breakout: price closes below S4 with volume confirmation
        if close_price < s4_level and vol_confirm:
            enter_short = True
        
        # Fade entries (mean reversion at extreme levels)
        # Long fade: price touches S3 with volume confirmation (only when not already long)
        if close_price <= s3_level and vol_confirm and position != 1:
            enter_long = True
        # Short fade: price touches R3 with volume confirmation (only when not already short)
        if close_price >= r3_level and vol_confirm and position != -1:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss, breaks below S3, or reaches R3 (profit target)
            exit_long = (close_price <= long_stop) or (close_price < s3_level) or (close_price >= r3_level)
        elif position == -1:
            # Exit short if price hits ATR stoploss, breaks above R3, or reaches S3 (profit target)
            exit_short = (close_price >= short_stop) or (close_price > r3_level) or (close_price <= s3_level)
        
        # Update stoploss levels when entering a position
        if enter_long:
            entry_price = close_price
            long_stop = entry_price - 2.0 * atr_14[i]
        elif enter_short:
            entry_price = close_price
            short_stop = entry_price + 2.0 * atr_14[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:
            # Trail long stop upward: max of current stop and (high - 2*ATR)
            long_stop = max(long_stop, high[i] - 2.0 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2*ATR)
            short_stop = min(short_stop, low[i] + 2.0 * atr_14[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals