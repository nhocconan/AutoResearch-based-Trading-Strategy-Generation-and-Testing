#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and ATR filter
# - Camarilla levels from 12h: L3 and H3 as key support/resistance
# - Long when price breaks above H3 with volume confirmation
# - Short when price breaks below L3 with volume confirmation
# - Volume confirmation: current 4h volume > 1.3x 20-period EMA
# - ATR filter: only trade when ATR(14) > 0.4 * ATR(50) to avoid low volatility
# - Exit: opposite Camarilla break (L3 for longs, H3 for shorts) or ATR trailing stop (2.0 * ATR)
# - Position sizing: 0.25 discrete level
# - Targets ~20-40 trades/year on 4h timeframe. Camarilla provides structure,
#   volume confirmation reduces fakeouts, ATR filter ensures sufficient volatility.
#   Works in bull/bear: breakouts capture strong moves, filters reduce whipsaws.

name = "4h_12h_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate ATR for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.3x 20-period EMA
    volume_ema_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_confirm = volume > 1.3 * volume_ema_20
    
    # ATR filter: only trade when short-term ATR > 0.4 * long-term ATR
    atr_filter = atr_14 > 0.4 * atr_50
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Actually standard Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # But we'll use the more common definition: H4/L4 for stronger levels
    # Standard Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # We'll use H3/L3 as they're more frequently tested
    hl_range_12h = high_12h - low_12h
    camarilla_h3_12h = close_12h + 1.1 * hl_range_12h / 2
    camarilla_l3_12h = close_12h - 1.1 * hl_range_12h / 2
    
    # Align Camarilla levels to 4h timeframe (using previous 12h bar's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3_12h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3_12h)
    
    # Track extremes for trailing stop
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (close[i] > camarilla_h3_aligned[i] and 
                     vol_confirm[i] and atr_filter[i])
        short_entry = (close[i] < camarilla_l3_aligned[i] and 
                      vol_confirm[i] and atr_filter[i])
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
                long_extreme = high[i]  # Initialize long extreme
            elif short_entry:
                position = -1
                signals[i] = -0.25
                short_extreme = low[i]  # Initialize short extreme
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update extremes
            if position == 1:  # Long position
                long_extreme = max(long_extreme, high[i])
                # ATR trailing stop: exit if price drops 2.0*ATR from extreme
                if close[i] < long_extreme - 2.0 * atr_14[i]:
                    position = 0
                    signals[i] = 0.0
                # Opposite Camarilla break (below L3)
                elif close[i] < camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                short_extreme = min(short_extreme, low[i])
                # ATR trailing stop: exit if price rises 2.0*ATR from extreme
                if close[i] > short_extreme + 2.0 * atr_14[i]:
                    position = 0
                    signals[i] = 0.0
                # Opposite Camarilla break (above H3)
                elif close[i] > camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals