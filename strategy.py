#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ADX regime filter
# - Uses 1h Camarilla pivot levels (H3/L3) for precise entry timing
# - Confirms with 4h volume > 1.5x 20-period average (institutional participation)
# - Filters by 1d ADX > 25 to ensure trending market (avoids choppy losses)
# - Exits when price touches opposite H4/L4 level or ATR-based stop (2.0x ATR)
# - Position size: 0.20 (20% of capital) for controlled risk
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Camarilla provides mathematically derived support/resistance that adapts to volatility

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
    tr_4h[0] = tr_4h[0] if len(tr_4h) > 0 else 0
    
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
    tr_1d[0] = tr_1d[0] if len(tr_1d) > 0 else 0
    
    # 1d ADX components
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di_14 = 100 * plus_dm_14 / np.where(tr_14 == 0, 1, tr_14)
    minus_di_14 = 100 * minus_dm_14 / np.where(tr_14 == 0, 1, tr_14)
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) == 0, 1, (plus_di_14 + minus_di_14))
    adx_1d = pd.Series(dx_14).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF indicators to 1h
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h.astype(float))
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
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
        if (np.isnan(volume_spike_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or atr_4h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Calculate 1h Camarilla pivots using previous bar's OHLC
        if i < 1:
            signals[i] = 0.0
            continue
            
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        prev_range = prev_high - prev_low
        
        if prev_range <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        h3 = prev_close + (prev_range * 1.1 / 4)
        l3 = prev_close - (prev_range * 1.1 / 4)
        h4 = prev_close + (prev_range * 1.1 / 2)
        l4 = prev_close - (prev_range * 1.1 / 2)
        
        if position == 1:  # Long position
            # Exit conditions: opposite Camarilla touch (L3) or ATR stoploss
            if low[i] <= l3:  # Touch L3 level
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Camarilla touch (H3) or ATR stoploss
            if high[i] >= h3:  # Touch H3 level
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and ADX filter
            if (high[i] >= h4 and  # Break above H4
                volume_spike_4h_aligned[i] and  # Volume confirmation
                adx_1d_aligned[i] > 25):  # Trending market (ADX > 25)
                position = 1
                entry_price = high[i]
                atr_stop = atr_4h_aligned[i]
                signals[i] = 0.20
            elif (low[i] <= l4 and   # Break below L4
                  volume_spike_4h_aligned[i] and  # Volume confirmation
                  adx_1d_aligned[i] > 25):  # Trending market (ADX > 25)
                position = -1
                entry_price = low[i]
                atr_stop = atr_4h_aligned[i]
                signals[i] = -0.20
    
    return signals