#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(9/21) crossover with 4h HMA(21) trend filter and volume confirmation.
# Long when 1h EMA(9) crosses above EMA(21) AND 4h HMA(21) trending up AND volume > 1.5x 20-period average.
# Short when 1h EMA(9) crosses below EMA(21) AND 4h HMA(21) trending down AND volume > 1.5x 20-period average.
# Exit on opposite EMA crossover or ATR-based stoploss (1.5*ATR from entry).
# Uses discrete position size 0.20. Designed to capture momentum moves with volume confirmation in trending markets.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: EMA(9) and EMA(21) ===
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # EMA crossover signals
    ema_cross_up = (ema_fast > ema_slow) & (np.roll(ema_fast, 1) <= np.roll(ema_slow, 1))
    ema_cross_down = (ema_fast < ema_slow) & (np.roll(ema_fast, 1) >= np.roll(ema_slow, 1))
    
    # === 4h Indicators: HMA(21) for trend ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 10  # 21/2 = 10.5 -> 10
    sqrt_len = 4   # sqrt(21) ≈ 4.58 -> 4
    wma_half = pd.Series(close_4h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_4h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_up = hma_4h_aligned > np.roll(hma_4h_aligned, 1)
    hma_down = hma_4h_aligned < np.roll(hma_4h_aligned, 1)
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike = volume > (1.5 * vol_ma_4h_aligned)
    
    # === 1h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h_raw = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 30 periods needed for EMA/HMA/ATR)
    warmup = 40
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_1h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if EMA crossover down
            if ema_cross_down[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if EMA crossover up
            if ema_cross_up[i]:
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
            # LONG: EMA crossover up AND HMA trending up AND volume spike
            if ema_cross_up[i] and hma_up[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: EMA crossover down AND HMA trending down AND volume spike
            elif ema_cross_down[i] and hma_down[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA9_21_4hHMA21_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0