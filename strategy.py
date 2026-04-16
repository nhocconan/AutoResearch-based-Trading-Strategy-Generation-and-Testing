#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 AND price > 12h EMA34 (uptrend) AND volume > 1.3x 20-period average.
# Short when price breaks below Camarilla S1 AND price < 12h EMA34 (downtrend) AND volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Camarilla levels provide precise intraday support/resistance from prior day's range.
# Volume confirmation ensures participation, 12h EMA34 aligns with higher timeframe trend.
# Designed to work in both bull (buy breakouts at R1) and bear (sell breakdowns at S1) markets.
# Target: 75-150 trades over 4 years (19-38/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Based on prior 4h bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no prior
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # === 4h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:  # Need enough for EMA34 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA34 for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA, 20 for volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r1_level = r1[i]
        s1_level = s1[i]
        ema_12h = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below S1 or volume spike ends
            if price < s1_level or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above R1 or volume spike ends
            if price > r1_level or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND price > 12h EMA34 (uptrend) AND volume spike
            if price > r1_level and price > ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND price < 12h EMA34 (downtrend) AND volume spike
            elif price < s1_level and price < ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_12hEMA34_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0