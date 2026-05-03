#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 1d Camarilla pivot levels (R1, S1) for breakout entries, aligned to 1d.
# Long when price breaks above R1 with volume > 1.8x 20-period MA and close > 1w EMA50 (uptrend).
# Short when price breaks below S1 with volume spike and close < 1w EMA50 (downtrend).
# Discrete sizing 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# Camarilla levels provide institutional support/resistance; 1w EMA50 filters counter-trend trades.
# Volume confirmation reduces false breakouts. Works in bull/bear via trend alignment.

name = "1d_Camarilla_R1S1_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1/12) = C + Range * 0.091666...
    # S1 = C - (Range * 1.1/12) = C - Range * 0.091666...
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    price_range = df_1d['high'] - df_1d['low']
    camarilla_r1 = typical_price + (price_range * 0.09166666666666666)
    camarilla_s1 = typical_price - (price_range * 0.09166666666666666)
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1d bar)
    # Note: Since primary timeframe is 1d, alignment is 1:1 but we still use the function for proper completed-bar timing
    df_1d = get_htf_data(prices, '1d')
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Volume regime: current 1d volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
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
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: break below S1 with volume spike in downtrend
            elif close_val < s1 and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below S1 OR trend turns down
            if close_val < s1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 OR trend turns up
            if close_val > r1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals