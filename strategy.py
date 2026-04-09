#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and daily trend filter
# - Uses 4h Camarilla pivot levels (H3/L3) for breakout signals on 1h timeframe
# - Confirms with 4h volume > 1.5x 20-period average (institutional participation)
# - Filters by 1d ADX > 25 to ensure trending market conditions
# - Exits when price touches opposite Camarilla level (H3/L3) or ATR-based stop (2.0x ATR)
# - Position size: 0.20 (20% of capital) to manage drawdown in volatile markets
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Camarilla pivots provide mathematical support/resistance that adapts to volatility

name = "1h_4h_1d_camarilla_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h True Range for ATR
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr_4h[0]
    
    # 4h ATR(14) for stoploss
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # 4h Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(14) for ADX calculation
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d +DM and -DM for ADX
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth DM and TR for ADX
    atr_14_smooth = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # 1d ADX(14)
    plus_di = 100 * (plus_dm_smooth / atr_14_smooth)
    minus_di = 100 * (minus_dm_smooth / atr_14_smooth)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_trend = adx_1d > 25  # trending market
    
    # Align 4h indicators to 1h
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h.astype(float))
    
    # Align 1d indicators to 1h
    adx_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_trend.astype(float))
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(volume_spike_4h_aligned[i]) or
            np.isnan(adx_trend_aligned[i]) or atr_4h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Calculate 1h Camarilla pivot levels using previous 4h bar's OHLC
        # We need the completed 4h bar that just closed
        # Find the index of the last completed 4h bar in 1h terms
        # Since 4h = 4 * 1h, we can use the 4h data aligned to 1h
        # But we need to calculate Camarilla from 4h OHLC
        
        # Get the 4h OHLC values aligned to current 1h bar
        # We'll use the 4h data that's already aligned via our HTF mechanism
        # However, we need to compute Camarilla levels from 4h OHLC
        
        # Instead, compute Camarilla levels using 4h data and align them
        # Calculate typical price for 4h Camarilla
        typical_price_4h = (high_4h + low_4h + close_4h) / 3
        range_4h = high_4h - low_4h
        
        # Camarilla levels: H3/H4/L3/L4
        camarilla_h3 = typical_price_4h + (range_4h * 1.1 / 4)
        camarilla_l3 = typical_price_4h - (range_4h * 1.1 / 4)
        camarilla_h4 = typical_price_4h + (range_4h * 1.1 / 2)
        camarilla_l4 = typical_price_4h - (range_4h * 1.1 / 2)
        
        # Align Camarilla levels to 1h
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
        
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Camarilla touch (L3) or ATR stoploss
            if low[i] <= camarilla_l3_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Camarilla touch (H3) or ATR stoploss
            if high[i] >= camarilla_h3_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and trend filter
            if (high[i] >= camarilla_h3_aligned[i] and  # Break above H3
                volume_spike_4h_aligned[i] and        # Volume confirmation
                adx_trend_aligned[i]):                # Trending market
                position = 1
                entry_price = high[i]
                atr_stop = atr_4h_aligned[i]
                signals[i] = 0.20
            elif (low[i] <= camarilla_l3_aligned[i] and   # Break below L3
                  volume_spike_4h_aligned[i] and        # Volume confirmation
                  adx_trend_aligned[i]):                # Trending market
                position = -1
                entry_price = low[i]
                atr_stop = atr_4h_aligned[i]
                signals[i] = -0.20
    
    return signals