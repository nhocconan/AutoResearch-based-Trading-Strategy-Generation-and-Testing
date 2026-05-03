#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# In bull regime (price > 1d EMA50), we go long on upper Donchian breakout with volume spike.
# In bear regime (price < 1d EMA50), we go short on lower Donchian breakout with volume spike.
# Uses ATR-based stoploss via signal=0 when price moves against position by 2.5*ATR.
# Designed for fewer trades (~30-50/year) to minimize fee drag while capturing strong trends.

name = "4h_Donchian20_1dTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Get current values
        highest_val = highest_20[i]
        lowest_val = lowest_20[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(highest_val) or np.isnan(lowest_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA50, bear if close < 1d EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Breakout conditions
        upper_breakout = close_val > highest_val
        lower_breakout = close_val < lowest_val
        
        # Regime-based entry conditions with volume confirmation
        if is_bull_regime:
            # Long: Upper Donchian breakout with volume spike
            long_entry = upper_breakout and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Lower Donchian breakout with volume spike
            short_entry = lower_breakout and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long position: trail stop or exit on breakdown
            stop_price = entry_price - 2.5 * atr_val
            if close_val < stop_price or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: trail stop or exit on breakout
            stop_price = entry_price + 2.5 * atr_val
            if close_val > stop_price or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals