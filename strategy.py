#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and ATR-based stoploss
# - Long: Price breaks above Camarilla R4 level (12h) + volume > 1.8x 20-period 12h average volume
# - Short: Price breaks below Camarilla S4 level (12h) + volume > 1.8x 20-period 12h average volume
# - Exit: ATR-based trailing stop (2.5 ATR from extreme) or opposite Camarilla breakout (R3/S3)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots from 12h timeframe provide institutional support/resistance levels
# - Volume confirmation filters out weak breakouts and increases signal quality
# - ATR stoploss manages risk during volatile periods while allowing trends to develop

name = "6h_12h_camarilla_breakout_volume_v2"
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
    
    # Load 12h data ONCE before loop for Camarilla pivots and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute Camarilla pivot levels on 12h timeframe (using previous 12h bar's OHLC)
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r4 = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    camarilla_r3 = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3 = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s4 = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Pre-compute ATR for stoploss (6h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        r4 = r4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above Camarilla R4 with volume confirmation
        if close_price > r4 and vol_confirm:
            enter_long = True
        
        # Short breakout: price closes below Camarilla S4 with volume confirmation
        if close_price < s4 and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss or breaks below Camarilla S3
            exit_long = (close_price <= long_stop) or (close_price < s3)
        elif position == -1:
            # Exit short if price hits ATR stoploss or breaks above Camarilla R3
            exit_short = (close_price >= short_stop) or (close_price > r3)
        
        # Update stoploss levels when entering a position
        if enter_long:
            entry_price = close_price
            long_stop = entry_price - 2.5 * atr_14[i]
        elif enter_short:
            entry_price = close_price
            short_stop = entry_price + 2.5 * atr_14[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:
            # Trail long stop upward: max of current stop and (high - 2.5*ATR)
            long_stop = max(long_stop, high[i] - 2.5 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2.5*ATR)
            short_stop = min(short_stop, low[i] + 2.5 * atr_14[i])
        
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