#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) with 1d EMA(50) trend filter and 1d volume spike confirmation.
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when Williams %R < -80 (oversold) AND price > EMA(50) (uptrend) AND volume > 1.8x 20-period 1d average
# Short when Williams %R > -20 (overbought) AND price < EMA(50) (downtrend) AND volume > 1.8x 20-period 1d average
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) or ATR-based stoploss (2.5*ATR from entry)
# Uses discrete position size 0.25. Designed to capture mean reversions in trending markets with volume confirmation.
# Works in both bull and bear markets by requiring trend (price vs EMA50) and volume confirmation, avoiding choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Williams %R (14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # === 1d Indicators: EMA(50) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    # === 12h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50/Williams %R)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_12h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        wr = williams_r[i]
        atr_val = atr_12h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (mean reversion complete)
            if wr > -50:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (mean reversion complete)
            if wr < -50:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > EMA(50) (uptrend) AND volume spike
            if wr < -80 and price > ema_50 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R > -20 (overbought) AND price < EMA(50) (downtrend) AND volume spike
            elif wr > -20 and price < ema_50 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_1dVolumeSpike_V1"
timeframe = "12h"
leverage = 1.0