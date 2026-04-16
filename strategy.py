#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and ATR stoploss.
# Long when price breaks above Camarilla R1 AND volume > 1.5x 20-period average AND ATR(14) < 0.03*price (low volatility).
# Short when price breaks below Camarilla S1 AND volume > 1.5x 20-period average AND ATR(14) < 0.03*price.
# Exit on ATR-based stoploss (1.5*ATR from entry) or opposite Camarilla level (S1 for long, R1 for short).
# Uses discrete position size 0.25. Designed to capture intraday momentum with volume confirmation in low-volatility regimes.
# Works in both bull and bear markets by requiring volume confirmation and volatility filter, avoiding false breakouts in choppy conditions.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (from previous day) ===
    # Camarilla levels calculated from previous day's OHLC
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # We need to use previous day's data, so shift by 1
    prev_day_close = pd.Series(close).shift(1).values
    prev_day_high = pd.Series(high).shift(1).values
    prev_day_low = pd.Series(low).shift(1).values
    camarilla_r1 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 12
    camarilla_s1 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 12
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h ATR for volatility filter and stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ratio = atr_4h_raw / close  # ATR as percentage of price
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for volume MA)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_ratio[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        vol_filter = atr_ratio[i] < 0.03  # Low volatility filter
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_4h_raw[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_4h_raw[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND low volatility
            if price > camarilla_r1[i] and vol_spike and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND low volatility
            elif price < camarilla_s1[i] and vol_spike and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_VolatilityFilter_V1"
timeframe = "4h"
leverage = 1.0