#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volume spike.
# Long when price < 4h VWAP - 1.0*ATR(4h) AND 1d volume > 2.0x 20-period average.
# Short when price > 4h VWAP + 1.0*ATR(4h) AND 1d volume > 2.0x 20-period average.
# Exit when price crosses 4h VWAP.
# Uses discrete position size 0.20. Session filter: 08-20 UTC.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.
# Works in bull/bear: mean reversion in range, trend filter avoids counter-trend in strong moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for VWAP and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 4h Indicators: VWAP ===
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vwap_num = np.cumsum(typical_price_4h * volume_4h)
    vwap_den = np.cumsum(volume_4h)
    vwap_4h = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price_4h)
    vwap_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # === 4h Indicators: ATR(14) ===
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume spike filter ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        vwap_val = vwap_aligned[i]
        atr_val = atr_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 2.0x 20-period average (using 1d volume MA)
        vol_filter = vol > 2.0 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses above VWAP
            if price >= vwap_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses below VWAP
            if price <= vwap_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price < VWAP - 1.0*ATR with volume confirmation
            if price < (vwap_val - 1.0 * atr_val) and vol_filter:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: price > VWAP + 1.0*ATR with volume confirmation
            elif price > (vwap_val + 1.0 * atr_val) and vol_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_VWAP_ATR_MeanReversion_1dVolumeSpike_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0