#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1w HMA(21) is rising AND 1d volume > 1.3x 20-period average.
# Short when price breaks below Donchian(20) low AND 1w HMA(21) is falling AND 1d volume > 1.3x 20-period average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Donchian break.
# Uses discrete position size 0.25. Designed to work in both bull and bear markets by requiring
# volume confirmation and using symmetric breakout levels with trend filter.
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1w Indicators: HMA(21) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(df_1w['close']).rolling(window=half_len, min_periods=half_len).mean().values
    wma_full = pd.Series(df_1w['close']).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_1w = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    # HMA direction: rising if current > previous
    hma_rising = hma_1w_aligned > np.concatenate([[hma_1w_aligned[0]], hma_1w_aligned[:-1]])
    hma_falling = hma_1w_aligned < np.concatenate([[hma_1w_aligned[0]], hma_1w_aligned[:-1]])
    
    # === 1d Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_1d)
    
    # === 1d ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 21 for HMA)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_1d_raw[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1d_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (opposite breakout)
            if price < low_roll[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (opposite breakout)
            if price > high_roll[i]:
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
            # LONG: Price breaks above Donchian high AND HMA rising AND volume spike
            if price > high_roll[i] and hma_rising[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND HMA falling AND volume spike
            elif price < low_roll[i] and hma_falling[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wHMA21_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0