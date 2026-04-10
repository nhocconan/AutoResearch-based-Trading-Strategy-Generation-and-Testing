#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and ATR volatility filter
# - Entry: Long when price breaks above Camarilla H3 (1d) + 1d volume > 2.0x 20-period average + ATR(14, 4h) > ATR(50, 4h) * 0.3
#          Short when price breaks below Camarilla L3 (1d) + same volume and volatility filters
# - Exit: Close-based reversal - exit long when price < Camarilla L3 (1d), exit short when price > Camarilla H3 (1d)
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Camarilla pivots from 1d provide intraday support/resistance levels that work in ranging and trending markets
# - Volume confirmation ensures participation, volatility filter avoids low-volatility false breakouts
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within HARD MAX: 400 total

name = "4h_1d_camarilla_vol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for Camarilla pivots and volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous 1d OHLC)
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H3 = pivot + (range * 1.1 / 2)
    # L3 = pivot - (range * 1.1 / 2)
    # H4 = pivot + (range * 1.1)
    # L4 = pivot - (range * 1.1)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    price_range = high_1d - low_1d
    camarilla_h3 = typical_price + (price_range * 1.1 / 2.0)
    camarilla_l3 = typical_price - (price_range * 1.1 / 2.0)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels for today's trading)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 4h ATR (14-period and 50-period for volatility regime)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_4h = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, prices, atr_14_4h)
    atr_50_aligned = align_htf_to_ltf(prices, prices, atr_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmation = volume_1d_current > 2.0 * volume_ma_aligned[i]
        
        # Volatility expansion filter: ATR(14) > 0.3 * ATR(50) indicates expanding volatility (genuine breakout)
        volatility_filter = atr_14_aligned[i] > 0.3 * atr_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume confirmation + volatility filter
            if (close_price > camarilla_h3_aligned[i] and 
                volume_confirmation and 
                volatility_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume confirmation + volatility filter
            elif (close_price < camarilla_l3_aligned[i] and 
                  volume_confirmation and 
                  volatility_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_aligned[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < camarilla_l3_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_aligned[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > camarilla_h3_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals