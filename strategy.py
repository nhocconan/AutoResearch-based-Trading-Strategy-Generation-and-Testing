#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: long when %R < -80 (oversold) and price > 1d EMA50 (uptrend),
# short when %R > -20 (overbought) and price < 1d EMA50 (downtrend).
# Volume must be > 1.3x 20-period 4h average for confirmation.
# Exit when Williams %R reverses (%R > -50 for long, %R < -50 for short) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean reversals in trending markets with volume filter.
# Works in both bull and bear markets by requiring trend alignment via 1d EMA50, avoiding counter-trend trades.
# Target: 100-200 total trades over 4 years (25-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 4h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_4h)
    
    # === 1d Indicators: EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    price_above_ema = close > ema_50_1d_aligned
    price_below_ema = close < ema_50_1d_aligned
    
    # === 4h Indicators: ATR for stoploss (14-period) ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50/W%R)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_4h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R reverses (above -50) or ATR stoploss hit
            if williams_r[i] > -50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R reverses (below -50) or ATR stoploss hit
            if williams_r[i] < -50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume spike
            if williams_r[i] < -80 and price_above_ema[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume spike
            elif williams_r[i] > -20 and price_below_ema[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0