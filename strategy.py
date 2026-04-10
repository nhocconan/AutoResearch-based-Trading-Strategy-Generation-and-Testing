#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR volatility filter
# - Entry: Long when price breaks above Camarilla H3 (1d) + 1d volume > 1.5x 20-period average + ATR(14, 1d) > ATR(50, 1d) * 0.6
#          Short when price breaks below Camarilla L3 (1d) + same volume and volatility filters
# - Exit: Close-based reversal - exit long when price < Camarilla L3 (1d), exit short when price > Camarilla H3 (1d)
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within HARD MAX: 400 total
# - Camarilla pivots from 1d provide key intraday support/resistance, volume spike confirms participation,
#   volatility expansion filter ensures we trade during genuine breakouts, not low-volatility squeezes
# - Works in bull markets via breakouts above H3, works in bear markets via shorting breakdowns below L3

name = "4h_1d_camarilla_vol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for Camarilla, volume and ATR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day's range)
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))  # for stoploss reference
    camarilla_l4 = np.zeros(len(close_1d))  # for stoploss reference
    
    for i in range(1, len(close_1d)):  # Start from 1 to have previous day's data
        # Previous day's high, low, close
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Camarilla levels calculation
        range_val = prev_high - prev_low
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
        camarilla_h4[i] = prev_close + range_val * 1.1 / 2
        camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
    # For first bar, use same day's data (will be refined as more data comes)
    camarilla_h3[0] = close_1d[0] + (high_1d[0] - low_1d[0]) * 1.1 / 4
    camarilla_l3[0] = close_1d[0] - (high_1d[0] - low_1d[0]) * 1.1 / 4
    camarilla_h4[0] = close_1d[0] + (high_1d[0] - low_1d[0]) * 1.1 / 2
    camarilla_l4[0] = close_1d[0] - (high_1d[0] - low_1d[0]) * 1.1 / 2
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period and 50-period for volatility regime)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align all HTF data to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, prices, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        # Volatility expansion filter: ATR(14) > 0.6 * ATR(50) indicates expanding volatility (genuine breakout)
        volatility_filter = atr_14_aligned[i] > 0.6 * atr_50_aligned[i]
        
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
                stop_loss = entry_price - 2.0 * atr_4h_aligned[i]
                # Exit conditions: price < Camarilla L3 level OR stoploss hit
                if close_price < camarilla_l3_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_4h_aligned[i]
                # Exit conditions: price > Camarilla H3 level OR stoploss hit
                if close_price > camarilla_h3_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals