#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and ATR-based stoploss.
# Long when price breaks above Camarilla R3 level AND volume > 1.5x 20-period 12h average.
# Short when price breaks below Camarilla S3 level AND volume > 1.5x 20-period 12h average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Camarilla break (R4/S4).
# Uses discrete position size 0.25. Designed to capture strong intraday momentum moves with volume confirmation.
# Works in both bull and bear markets by requiring volume confirmation and avoiding false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (from prior 6h bar) ===
    # Camarilla levels calculated from prior bar's OHLC
    # R4 = close + ((high - low) * 1.5)
    # R3 = close + ((high - low) * 1.25)
    # S3 = close - ((high - low) * 1.25)
    # S4 = close - ((high - low) * 1.5)
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    camarilla_r4 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        camarilla_r3[i] = prev_close + ((prev_high - prev_low) * 1.25)
        camarilla_s3[i] = prev_close - ((prev_high - prev_low) * 1.25)
        camarilla_r4[i] = prev_close + ((prev_high - prev_low) * 1.5)
        camarilla_s4[i] = prev_close - ((prev_high - prev_low) * 1.5)
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for volume MA/ATR)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_r4[i]) or
            np.isnan(camarilla_s4[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_6h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks above Camarilla R4 (take profit) OR below S3 (stoploss)
            if price > camarilla_r4[i] or price < camarilla_s3[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks below Camarilla S4 (take profit) OR above R3 (stoploss)
            if price < camarilla_s4[i] or price > camarilla_r3[i]:
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
            # LONG: Price breaks above Camarilla R3 AND volume spike
            if price > camarilla_r3[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike
            elif price < camarilla_s3[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "6h"
leverage = 1.0