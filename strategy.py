#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Uses 4h Camarilla pivot levels (R1, S1) for breakout entries, aligned to 1h.
# Long when price breaks above R1 with volume > 1.5x 20-period MA and close > 4h EMA50 (uptrend).
# Short when price breaks below S1 with volume spike and close < 4h EMA50 (downtrend).
# Discrete sizing 0.20. Session filter 08-20 UTC to reduce noise.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Camarilla levels provide institutional support/resistance; EMA50 filters counter-trend trades.
# Volume confirmation reduces false breakouts. Works in bull/bear via trend alignment.

name = "1h_Camarilla_R1S1_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and EMA50
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla pivot levels
    # Camarilla: Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1/12) = C + Range * 0.0916667
    # S1 = C - (Range * 1.1/12) = C - Range * 0.0916667
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    price_range = df_4h['high'] - df_4h['low']
    camarilla_r1 = typical_price + (price_range * 0.0916667)
    camarilla_s1 = typical_price - (price_range * 0.0916667)
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1.values)
    
    # Volume regime: current 1h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above R1 with volume spike in uptrend
            if close_val > r1 and vol_spike and is_uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            # Short: break below S1 with volume spike in downtrend
            elif close_val < s1 and vol_spike and is_downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below S1 OR trend turns down
            if close_val < s1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 OR trend turns up
            if close_val > r1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals