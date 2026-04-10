#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume spike and 12h ADX regime filter
# - Entry: Long when price breaks above 6h Camarilla H3 + 12h volume > 2.0x 20-period average + 12h ADX > 25 (trending regime)
#          Short when price breaks below 6h Camarilla L3 + 12h volume > 2.0x 20-period average + 12h ADX > 25 (trending regime)
# - Exit: Close-based reversal - exit long when price < 6h Camarilla L3, exit short when price > 6h Camarilla H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 6h
# - Position sizing: 0.25 (discrete level)
# - Uses 6h price structure for entries/exits, 12h volume for participation confirmation,
#   and 12h ADX to filter for trending markets where breakouts work best
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within HARD MAX: 300 total
# - Camarilla pivots identify key intraday levels, volume confirms institutional interest,
#   ADX>25 ensures we only trade in trending markets reducing false breakouts
# - Works in bull markets via breakouts and in bear markets via short breakdowns with volume confirmation

name = "6h_12h_camarilla_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 12h data for volume and ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 6h Camarilla levels (based on previous bar's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    # We use previous bar's values to avoid look-ahead
    prev_high_6h = np.roll(high_6h, 1)
    prev_low_6h = np.roll(low_6h, 1)
    prev_close_6h = np.roll(close_6h, 1)
    prev_high_6h[0] = high_6h[0]  # First bar uses current values (no previous)
    prev_low_6h[0] = low_6h[0]
    prev_close_6h[0] = close_6h[0]
    
    camarilla_h3 = prev_close_6h + 1.0 * (prev_high_6h - prev_low_6h)
    camarilla_l3 = prev_close_6h - 1.0 * (prev_high_6h - prev_low_6h)
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = -np.diff(low_12h, prepend=low_12h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    alpha = 1.0 / 14
    atr_12h = np.zeros_like(tr)
    plus_dm_12h = np.zeros_like(plus_dm)
    minus_dm_12h = np.zeros_like(minus_dm)
    
    atr_12h[0] = tr[0]
    plus_dm_12h[0] = plus_dm[0]
    minus_dm_12h[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        atr_12h[i] = (1 - alpha) * atr_12h[i-1] + alpha * tr[i]
        plus_dm_12h[i] = (1 - alpha) * plus_dm_12h[i-1] + alpha * plus_dm[i]
        minus_dm_12h[i] = (1 - alpha) * minus_dm_12h[i-1] + alpha * minus_dm[i]
    
    # Avoid division by zero
    plus_di_12h = np.where(atr_12h > 0, 100 * plus_dm_12h / atr_12h, 0.0)
    minus_di_12h = np.where(atr_12h > 0, 100 * minus_dm_12h / atr_12h, 0.0)
    
    dx_12h = np.where((plus_di_12h + minus_di_12h) > 0, 
                     100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 
                     0.0)
    
    # Smoothed DX to get ADX
    adx_12h = np.zeros_like(dx_12h)
    adx_12h[13] = np.mean(dx_12h[14:28]) if len(dx_12h) >= 28 else np.mean(dx_12h[14:]) if len(dx_12h) > 14 else 0.0
    for i in range(14, len(dx_12h)):
        adx_12h[i] = (1 - alpha) * adx_12h[i-1] + alpha * dx_12h[i]
    
    # Calculate 6h ATR (14-period) for stoploss
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = 0
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, prices, camarilla_h3)  # 6h data already aligned
    camarilla_l3_aligned = align_htf_to_ltf(prices, prices, camarilla_l3)  # 6h data already aligned
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    atr_6h_aligned = align_htf_to_ltf(prices, prices, atr_6h)  # 6h data already aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close
        close_price = close_6h[i]
        
        # Get current 12h volume for confirmation
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirmation = volume_12h_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        # ADX filter: > 25 indicates trending market (good for breakouts)
        adx_filter = adx_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume confirmation + trending market
            if (close_price > camarilla_h3_aligned[i] and 
                volume_confirmation and 
                adx_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume confirmation + trending market
            elif (close_price < camarilla_l3_aligned[i] and 
                  volume_confirmation and 
                  adx_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_6h_aligned[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < camarilla_l3_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_6h_aligned[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > camarilla_h3_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals