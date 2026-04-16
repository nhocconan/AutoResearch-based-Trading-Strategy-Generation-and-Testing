#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1d EMA(34) trend filter and 1d volume spike confirmation.
# Long when Williams %R crosses above -80 (oversold reversal) AND price > 1d EMA(34) AND volume > 1.5x 20-period 1d average.
# Short when Williams %R crosses below -20 (overbought reversal) AND price < 1d EMA(34) AND volume > 1.5x 20-period 1d average.
# Exit when Williams %R crosses -50 (mean reversion) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean reversals in established trends.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.
# Works in both bull and bear markets by requiring trend alignment (price vs EMA) and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 1d Indicators: EMA(34) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 6h ATR for stoploss
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        is_uptrend = price > ema_34_1d_aligned[i]
        is_downtrend = price < ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (mean reversion)
            if wr > -50 and wr_prev <= -50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (mean reversion)
            if wr < -50 and wr_prev >= -50:
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
            # LONG: Williams %R crosses above -80 (oversold reversal) AND uptrend AND volume spike
            if wr > -80 and wr_prev <= -80 and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 (overbought reversal) AND downtrend AND volume spike
            elif wr < -20 and wr_prev >= -20 and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR14_1dEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0