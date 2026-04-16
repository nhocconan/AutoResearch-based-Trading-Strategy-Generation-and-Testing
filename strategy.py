#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low.
# Long when Bull Power > 0 AND Bear Power < 0 (both positive/negative) AND price > 1d EMA34 AND volume > 1.2x 20-period average.
# Short when Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA34 AND volume > 1.2x 20-period average.
# Uses discrete position size 0.25. Works in both bull and bear markets by requiring
# alignment with higher timeframe trend and volume confirmation to filter false signals.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray Index (EMA13) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # === 1d Indicators: EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.2x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for 1d EMA34)
    warmup = 50
    
    # Track position state and entry price for potential stoploss (though we rely on signal reversal)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        ema_trend = ema34_1d_aligned[i]
        
        # === EXIT LOGIC: Flat on signal reversal or loss of conditions ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Elder Ray conditions no longer met for long
            if not (bull_power[i] > 0 and bear_power[i] < 0 and price > ema_trend and vol_spike):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Elder Ray conditions no longer met for short
            if not (bear_power[i] > 0 and bull_power[i] < 0 and price < ema_trend and vol_spike):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA34 AND volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and price > ema_trend and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA34 AND volume spike
            elif (bear_power[i] > 0 and bull_power[i] < 0 and price < ema_trend and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            # Hold current position
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0