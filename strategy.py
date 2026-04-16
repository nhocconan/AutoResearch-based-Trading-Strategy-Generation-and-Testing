#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1d EMA(34) trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA(34) AND volume > 1.5x 20-period average.
# Short when Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA(34) AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses -50 (mean reversion) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean reversion in trending markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# Works in both bull and bear markets by requiring trend alignment via 1d EMA(34).

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
    
    # === Volume Confirmation: volume > 1.5x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(atr_6h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        wr = williams_r[i]
        vol_conf = volume_confirm[i]
        atr_val = atr_6h[i]
        price_vs_ema = price > ema_34_1d_aligned[i]
        
        # Williams %R crossover signals
        wr_cross_above_80 = wr > -80 and williams_r[i-1] <= -80
        wr_cross_below_20 = wr < -20 and williams_r[i-1] >= -20
        wr_cross_above_50 = wr > -50 and williams_r[i-1] <= -50
        wr_cross_below_50 = wr < -50 and williams_r[i-1] >= -50
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (overbought)
            if wr_cross_above_50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (oversold)
            if wr_cross_below_50:
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
            # LONG: Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA(34) AND volume confirmation
            if wr_cross_above_80 and price_vs_ema and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA(34) AND volume confirmation
            elif wr_cross_below_20 and not price_vs_ema and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR14_1dEMA34_VolumeConfirm_V1"
timeframe = "6h"
leverage = 1.0