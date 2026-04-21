#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeRegime
Hypothesis: 1h Camarilla pivot (R1/S1) breakouts filtered by 4h EMA34 trend and volume regime (ATR-based).
Enter long when price breaks above 1h R1 with 4h uptrend and elevated volatility.
Enter short when price breaks below 1h S1 with 4h downtrend and elevated volatility.
Exit on ATR(14) trailing stop (2.0*ATR) or opposite level break.
Uses 4h trend for signal direction, 1h only for entry timing to reduce noise.
Target: 15-35 trades/year to minimize fee drag on 1h timeframe.
Works in bull/bear via 4h trend alignment and volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend, 1h is primary)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # === 1h Camarilla Pivot Levels (R1, S1) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Need previous completed 1h bar for pivot calculation
    # We'll use rolling window on 1h data
    high_1h = pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values  # previous bar high
    low_1h = pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values    # previous bar low
    close_1h = pd.Series(close).rolling(window=2, min_periods=2).last().shift(1).values # previous bar close
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1h - low_1h) * 1.1 / 12.0
    r1_1h = close_1h + camarilla_range
    s1_1h = close_1h - camarilla_range
    
    # === 4h EMA34 for HTF trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === ATR (14-period) for volatility regime and stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: ATR ratio (current / 50-period average)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr / atr_ma  # >1.0 = elevated volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) 
            or np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr[i]) 
            or np.isnan(vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume regime: elevated volatility (ATR ratio > 1.2)
            vol_filter = vol_regime[i] > 1.2
            
            # Long conditions: price > 1h R1, 4h uptrend, elevated volatility
            long_breakout = price > r1_1h[i]
            long_trend = price > ema_34_4h_aligned[i]
            
            # Short conditions: price < 1h S1, 4h downtrend, elevated volatility
            short_breakout = price < s1_1h[i]
            short_trend = price < ema_34_4h_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 1h S1 (support broken)
            elif price < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 1h R1 (resistance broken)
            elif price > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeRegime"
timeframe = "1h"
leverage = 1.0