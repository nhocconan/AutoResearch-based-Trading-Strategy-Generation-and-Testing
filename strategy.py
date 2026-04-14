#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Bollinger Band breakout with weekly Keltner channel filter
# Long when price breaks above BB upper band AND is above KC upper band (strong bullish momentum)
# Short when price breaks below BB lower band AND is below KC lower band (strong bearish momentum)
# Exit when price crosses the Bollinger middle band (mean reversion)
# Bollinger Bands identify volatility expansion/contraction, Keltner Channel confirms trend strength
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee dust while capturing momentum bursts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1w data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Bollinger Bands on 12h: 20-period, 2 std dev
    close_12h = df_12h['close'].values
    bb_middle = pd.Series(close_12h).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(close_12h).rolling(window=20, min_periods=20).std()
    bb_upper = (bb_middle + 2 * bb_std).values
    bb_lower = (bb_middle - 2 * bb_std).values
    
    # Calculate Keltner Channel on 1w: 20-period EMA, ATR(10) * 1.5
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    kc_middle = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean()
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(abs(low_1w - pd.Series(close_1w).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    kc_upper = (kc_middle + 1.5 * atr).values
    kc_lower = (kc_middle - 1.5 * atr).values
    
    # Align indicators to 12h timeframe
    bb_middle_aligned = align_htf_to_ltf(prices, df_12h, bb_middle.values)
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    kc_upper_aligned = align_htf_to_ltf(prices, df_1w, kc_upper, additional_delay_bars=1)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1w, kc_lower, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_middle_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(kc_upper_aligned[i]) or 
            np.isnan(kc_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price above BB upper AND above KC upper (strong bullish momentum)
            if price > bb_upper_aligned[i] and price > kc_upper_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short setup: price below BB lower AND below KC lower (strong bearish momentum)
            elif price < bb_lower_aligned[i] and price < kc_lower_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below BB middle (mean reversion)
            if price < bb_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above BB middle (mean reversion)
            if price > bb_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Bollinger_Keltner_Momentum"
timeframe = "12h"
leverage = 1.0