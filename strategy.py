#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Elder Ray confirmation and 1w trend filter
# - Williams Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND 1w close > 1w EMA34
# - Short when: Jaw > Teeth > Lips (bearish alignment) AND Bear Power < 0 AND 1w close < 1w EMA34
# - Exit when: Alligator lines cross (Lips crosses Teeth in opposite direction) OR ATR-based stoploss
# - Uses 1w trend filter to avoid counter-trend trades and ATR stoploss for risk control
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years)
# - Works in both bull and bear markets by aligning with 1w trend and using momentum confirmation

name = "12h_1w_alligator_elderray_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for stoploss
    
    # Pre-compute aligned 1w data
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    h_1w_aligned = align_htf_to_ltf(prices, df_1w, h_1w)
    l_1w_aligned = align_htf_to_ltf(prices, df_1w, l_1w)
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    # Pre-compute 1w EMA(34) for trend filter
    ema34_1w = pd.Series(c_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Pre-compute Williams Alligator components
    # Jaw: EMA13 shifted by 8 bars
    jaw = pd.Series(prices['close']).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    # Teeth: EMA8 shifted by 5 bars
    teeth = pd.Series(prices['close']).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    # Lips: EMA5 shifted by 3 bars
    lips = pd.Series(prices['close']).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Pre-compute Elder Ray components (using EMA13)
    ema13 = pd.Series(prices['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'] - ema13  # High - EMA13
    bear_power = prices['low'] - ema13   # Low - EMA13
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(h_1w_aligned[i]) or np.isnan(l_1w_aligned[i]) or 
            np.isnan(c_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1w bar values for trend filter
        # Since 12h timeframe, 1w data updates every 14 bars (168h/12h = 14)
        # Look back to the previous multiple of 14 to get completed 1w bar
        lookback_idx = (i // 14) * 14  # Start of current 1w bar
        if lookback_idx >= 14:  # Need at least one previous completed 1w bar
            prev_1w_idx = lookback_idx - 14  # Previous completed 1w bar
            
            if prev_1w_idx >= 0:
                pwc = c_1w_aligned[prev_1w_idx]  # Previous 1w close
                pema34 = ema34_1w_aligned[prev_1w_idx]  # Previous 1w EMA34
                
                # Williams Alligator conditions
                bullish_alignment = (lips[i] > teeth[i] > jaw[i])
                bearish_alignment = (jaw[i] > teeth[i] > lips[i])
                
                if position == 0:  # Flat - look for new entries
                    # Long entry: bullish alignment AND bull power > 0 AND 1w uptrend
                    if (bullish_alignment and 
                        bull_power[i] > 0 and 
                        pwc > pema34):
                        position = 1
                        entry_price = prices['close'].iloc[i]
                        signals[i] = 0.25
                    # Short entry: bearish alignment AND bear power < 0 AND 1w downtrend
                    elif (bearish_alignment and 
                          bear_power[i] < 0 and 
                          pwc < pema34):
                        position = -1
                        entry_price = prices['close'].iloc[i]
                        signals[i] = -0.25
                else:  # Have position - look for exit
                    # Exit conditions:
                    # 1. Alligator lines cross (Lips crosses Teeth in opposite direction)
                    # 2. ATR-based stoploss hit
                    exit_signal = False
                    if position == 1:  # Long position
                        # Exit long if lips crosses below teeth (end of bullish alignment)
                        if lips[i] < teeth[i]:
                            exit_signal = True
                        # ATR stoploss
                        elif prices['close'].iloc[i] < entry_price - 2.5 * atr[i]:
                            exit_signal = True
                    elif position == -1:  # Short position
                        # Exit short if lips crosses above teeth (end of bearish alignment)
                        if lips[i] > teeth[i]:
                            exit_signal = True
                        # ATR stoploss
                        elif prices['close'].iloc[i] > entry_price + 2.5 * atr[i]:
                            exit_signal = True
                    
                    if exit_signal:
                        position = 0
                        entry_price = 0.0
                        signals[i] = 0.0
                    else:
                        if position == 1:
                            signals[i] = 0.25
                        else:
                            signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals