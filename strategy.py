#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h/1d trend filter and volume confirmation.
# Long when 1h EMA(9) crosses above EMA(21) AND price > 4h EMA(34) AND 1d close > 1d EMA(50) AND volume > 1.2x 20-period average.
# Short when 1h EMA(9) crosses below EMA(21) AND price < 4h EMA(34) AND 1d close < 1d EMA(50) AND volume > 1.2x 20-period average.
# Exit on opposite EMA crossover or ATR-based stop (1.5*ATR).
# Uses discrete position size 0.20. Designed to capture strong trends with multi-timeframe alignment and volume confirmation.
# Session filter 08-20 UTC to avoid low-liquidity periods. Target: 60-150 total trades over 4 years (15-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: EMA(9), EMA(21), ATR(14) ===
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 4h Indicators: EMA(34) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === 1d Indicators: EMA(50) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike: volume > 1.2x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.2 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema34_4h_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema9_prev = ema9[i-1]
        ema21_prev = ema21[i-1]
        ema9_curr = ema9[i]
        ema21_curr = ema21[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if EMA(9) crosses below EMA(21)
            if ema9_prev >= ema21_prev and ema9_curr < ema21_curr:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if EMA(9) crosses above EMA(21)
            if ema9_prev <= ema21_prev and ema9_curr > ema21_curr:
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
            # EMA crossover signals
            bullish_cross = ema9_prev <= ema21_prev and ema9_curr > ema21_curr
            bearish_cross = ema9_prev >= ema21_prev and ema9_curr < ema21_curr
            
            # LONG: Bullish EMA crossover AND price > 4h EMA(34) AND 1d close > 1d EMA(50) AND volume spike
            if bullish_cross and price > ema34_4h_aligned[i] and close[i] > ema50_1d_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Bearish EMA crossover AND price < 4h EMA(34) AND 1d close < 1d EMA(50) AND volume spike
            elif bearish_cross and price < ema34_4h_aligned[i] and close[i] < ema50_1d_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA9_21_4hEMA34_1dEMA50_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0