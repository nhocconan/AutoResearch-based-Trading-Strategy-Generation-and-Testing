#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d volume spike and 1w EMA trend filter.
# Long when 6h Williams %R crosses above -80 (oversold bounce) AND 1d volume > 1.5x 20-period average AND price > 1w EMA50 (uptrend).
# Short when 6h Williams %R crosses below -20 (overbought rejection) AND 1d volume > 1.5x 20-period average AND price < 1w EMA50 (downtrend).
# Exit when Williams %R crosses -50 (mean reversion midpoint) or ATR-based stop (1.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture reversals in range-bound and trending markets.
# Target: 80-180 total trades over 4 years (20-45/year) to balance edge and fee drag.
# Works in both bull and bear markets: longs in uptrend (price>1w EMA50), shorts in downtrend (price<1w EMA50).

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
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1w Indicators: EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    price_above_ema = close > ema_50_1w_aligned
    price_below_ema = close < ema_50_1w_aligned
    
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
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for 1w EMA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or np.isnan(volume_spike[i]) or
            np.isnan(price_above_ema[i]) or np.isnan(price_below_ema[i]) or np.isnan(atr_6h[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h[i]
        
        # Williams %R crossover signals
        wr_cross_above_80 = williams_r[i] > -80 and williams_r[i-1] <= -80
        wr_cross_below_20 = williams_r[i] < -20 and williams_r[i-1] >= -20
        wr_cross_above_50 = williams_r[i] > -50 and williams_r[i-1] <= -50
        wr_cross_below_50 = williams_r[i] < -50 and williams_r[i-1] >= -50
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (overbought reversal)
            if wr_cross_above_50:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (oversold reversal)
            if wr_cross_below_50:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R crosses above -80 AND volume spike AND price > 1w EMA50 (uptrend)
            if wr_cross_above_80 and vol_spike and price_above_ema[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 AND volume spike AND price < 1w EMA50 (downtrend)
            elif wr_cross_below_20 and vol_spike and price_below_ema[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_1dVolumeSpike_1wEMA50_V1"
timeframe = "6h"
leverage = 1.0