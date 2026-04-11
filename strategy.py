#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and ATR stoploss
# - Long: Price breaks above Camarilla H3 level (12h) + volume > 1.5x 20-period 12h average volume
# - Short: Price breaks below Camarilla L3 level (12h) + volume > 1.5x 20-period 12h average volume
# - Exit: ATR-based trailing stop (2.0 ATR from extreme) or opposite Camarilla level break
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide mathematical support/resistance levels that work in ranging and trending markets
# - Volume confirmation filters out false breakouts and increases signal quality
# - ATR stoploss manages risk during volatile periods while allowing trends to develop

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop for Camarilla pivots and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla formulas: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4, L4 = C - (H-L)*1.1/2
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla levels for each 12h bar (using previous bar's OHLC)
    camarilla_h3 = np.full(len(close_12h), np.nan)
    camarilla_l3 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        h = high_12h[i-1]
        l = low_12h[i-1]
        c = close_12h[i-1]
        camarilla_h3[i] = c + (h - l) * 1.1 / 4
        camarilla_l3[i] = c - (h - l) * 1.1 / 4
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above H3 with volume confirmation
        if close_price > h3_level and vol_confirm:
            enter_long = True
        
        # Short breakout: price closes below L3 with volume confirmation
        if close_price < l3_level and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss or breaks below L3
            exit_long = (close_price <= long_stop) or (close_price < l3_level)
        elif position == -1:
            # Exit short if price hits ATR stoploss or breaks above H3
            exit_short = (close_price >= short_stop) or (close_price > h3_level)
        
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