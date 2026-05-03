#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1w EMA200 trend filter and volume confirmation.
# Long when price breaks above R4 (camarilla resistance 4) in bull trend (close > 1w EMA200) with volume spike.
# Short when price breaks below S4 (camarilla support 4) in bear trend (close < 1w EMA200) with volume spike.
# Uses discrete position sizing (0.30) to minimize fee churn.
# Camarilla levels derived from prior 1d candle (HLC) provide institutional pivot points.
# 1w EMA200 ensures alignment with higher timeframe trend. Volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R4S4_Breakout_1wEMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 5 or len(df_1w) < 200:  # Need sufficient data
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Camarilla levels from prior 1d candle (HLC)
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R4 = CP + (H-L)*1.1, S4 = CP - (H-L)*1.1
    camarilla_pivot = typical_price.values
    camarilla_r4 = camarilla_pivot + (range_val * 1.1)
    camarilla_s4 = camarilla_pivot - (range_val * 1.1)
    
    # Align Camarilla levels to 12h timeframe (use prior completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any value is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_200_1w_aligned[i]
        r4_level = camarilla_r4_aligned[i]
        s4_level = camarilla_s4_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions
        breakout_long = close_val > r4_level
        breakout_short = close_val < s4_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_long and vol_spike:
                signals[i] = 0.30
                position = 1
            elif is_bear_trend and breakout_short and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below S4 OR trend reversal
            if close_val < s4_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above R4 OR trend reversal
            if close_val > r4_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals