#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Keltner Channel breakout with 1w EMA50 trend filter and volume confirmation.
# Keltner Channel identifies volatility-based breakouts; breakouts in weekly trend direction with volume surge
# capture strong momentum moves. Works in bull/bear markets by filtering false breakouts in ranging conditions.
# Target: 20-50 trades/year per symbol.
name = "4h_Keltner_EMA50w_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Keltner Channel (20, 2) on 4h
    kc_period = 20
    kc_mult = 2
    atr_period = 20
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema_20 = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    upper_kc = ema_20 + (kc_mult * atr)
    lower_kc = ema_20 - (kc_mult * atr)
    
    # Align 1w EMA50 to 4h
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_period, atr_period, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20[i]) or np.isnan(atr[i]) or np.isnan(upper_kc[i]) or 
            np.isnan(lower_kc[i]) or np.isnan(ema_50w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_kc[i]
        lower = lower_kc[i]
        ema_50w = ema_50w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper  # Price breaks above upper Keltner
        bearish_breakout = price < lower  # Price breaks below lower Keltner
        
        if position == 0:
            # Look for entry in direction of weekly trend
            if bullish_breakout and (price > ema_50w) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and (price < ema_50w) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to middle line (EMA20)
            if price < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle line
            if price > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals