#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume spike and 1w ADX regime filter
# - Entry: Long when price breaks above 1d Camarilla H3 + 1w volume > 2.0x 20-period average + 1w ADX > 25 (trending regime)
#          Short when price breaks below 1d Camarilla L3 + 1w volume > 2.0x 20-period average + 1w ADX > 25 (trending regime)
# - Exit: Close-based reversal - exit long when price < 1d Camarilla L3, exit short when price > 1d Camarilla H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 1d
# - Position sizing: 0.25 (discrete level)
# - Uses 1d price structure for entries/exits, weekly volume for participation confirmation,
#   and weekly ADX to filter for trending markets where breakouts work best
# - Target: 50-100 total trades over 4 years (12-25/year) to stay within HARD MAX: 150 total
# - Camarilla pivots identify key intraday levels, volume confirms institutional interest,
#   ADX>25 ensures we only trade in trending markets reducing false breakouts
# - Works in bull markets via breakouts and in bear markets via short breakdowns with volume confirmation

name = "1d_1w_camarilla_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Pre-compute 1w data for volume and ADX
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    # We use previous day's values to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]  # First bar uses current values (no previous)
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_h3 = prev_close_1d + 1.0 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.0 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    alpha = 1.0 / 14
    atr_1w = np.zeros_like(tr)
    plus_dm_1w = np.zeros_like(plus_dm)
    minus_dm_1w = np.zeros_like(minus_dm)
    
    atr_1w[0] = tr[0]
    plus_dm_1w[0] = plus_dm[0]
    minus_dm_1w[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        atr_1w[i] = (1 - alpha) * atr_1w[i-1] + alpha * tr[i]
        plus_dm_1w[i] = (1 - alpha) * plus_dm_1w[i-1] + alpha * plus_dm[i]
        minus_dm_1w[i] = (1 - alpha) * minus_dm_1w[i-1] + alpha * minus_dm[i]
    
    # Avoid division by zero
    plus_di_1w = np.where(atr_1w > 0, 100 * plus_dm_1w / atr_1w, 0.0)
    minus_di_1w = np.where(atr_1w > 0, 100 * minus_dm_1w / atr_1w, 0.0)
    
    dx_1w = np.where((plus_di_1w + minus_di_1w) > 0, 
                     100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 
                     0.0)
    
    # Smoothed DX to get ADX
    adx_1w = np.zeros_like(dx_1w)
    adx_1w[13] = np.mean(dx_1w[14:28]) if len(dx_1w) >= 28 else np.mean(dx_1w[14:]) if len(dx_1w) > 14 else 0.0
    for i in range(14, len(dx_1w)):
        adx_1w[i] = (1 - alpha) * adx_1w[i-1] + alpha * dx_1w[i]
    
    # Calculate 1d ATR (14-period) for stoploss
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, prices, camarilla_h3)  # 1d data already aligned
    camarilla_l3_aligned = align_htf_to_ltf(prices, prices, camarilla_l3)  # 1d data already aligned
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, prices, atr_1d)  # 1d data already aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d close
        close_price = close_1d[i]
        
        # Get current 1w volume for confirmation
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirmation = volume_1w_aligned[i] > 2.0 * volume_ma_aligned[i]
        
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
                stop_loss = entry_price - 2.0 * atr_1d_aligned[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < camarilla_l3_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_1d_aligned[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > camarilla_h3_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals