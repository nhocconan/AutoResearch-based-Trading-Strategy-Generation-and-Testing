#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long: price breaks above Camarilla H3 level (1d), volume > 1.3x 20-period avg, 12h chop < 61.8 (trending)
# - Short: price breaks below Camarilla L3 level (1d), volume > 1.3x 20-period avg, 12h chop < 61.8 (trending)
# - Exit: price returns to Camarilla H4/L4 levels or ATR-based stop (2.0x ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in 12h timeframe with volume confirmation and trend filter

name = "12h_1d_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low), etc.
    # We use H3/L3 for entry and H4/L4 for exit
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate pivot levels for each 1d bar
    camarilla_h3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 12h chop regime filter (14-period)
    # Chop = 100 * log10(sum(atr14) / log10(highest-highest-lowest-lowest)) / log10(14)
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_12h[0] = high[0] - low[0]
    atr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high - lowest_low
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr_14 * 14) / np.log10(range_14)
    # Handle invalid values
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss
    atr_20 = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or np.isnan(chop[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Trend filter: 12h chop < 61.8 (indicates trending market, not ranging)
        chop_trend = chop[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla H3, volume confirmation, trending
        if close_price > h3 and vol_confirm and chop_trend:
            enter_long = True
        
        # Short breakout: price below Camarilla L3, volume confirmation, trending
        if close_price < l3 and vol_confirm and chop_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to H4/L4 or ATR-based stop
            exit_long = (close_price >= h4) or (close_price <= l4) or (close_price <= entry_price - 2.0 * atr_20[i])
        elif position == -1:
            # Exit short if price returns to H4/L4 or ATR-based stop
            exit_short = (close_price >= h4) or (close_price <= l4) or (close_price >= entry_price + 2.0 * atr_20[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
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