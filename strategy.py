#!/usr/bin/env python3
"""
4h_1d_CCI_BullBear_Divergence_Trend
Hypothesis: Use CCI(20) on daily timeframe to identify bullish/bearish divergences.
In bull markets: Buy when CCI forms higher low while price forms lower low (bullish divergence) with volume confirmation.
In bear markets: Sell when CCI forms lower high while price forms higher high (bearish divergence) with volume confirmation.
Uses 4h timeframe for execution with 1d CCI divergence signals, reducing trade frequency while capturing major reversals.
Works in both bull and bear markets by adapting to trend context via price action relative to 200-period SMA.
"""

name = "4h_1d_CCI_BullBear_Divergence_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma)
    
    # Get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate CCI(20) on daily timeframe
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(typical_price).rolling(window=20, min_periods=20).std().values
    # Avoid division by zero
    tp_std = np.where(tp_std == 0, 1e-10, tp_std)
    cci = (typical_price - tp_ma) / (0.015 * tp_std)
    
    # Identify bullish and bearish divergences
    # Bullish divergence: price makes lower low, CCI makes higher low
    # Bearish divergence: price makes higher high, CCI makes lower high
    
    # Find price swing points (simplified: local minima/maxima over 3 periods)
    price_lows = np.zeros(len(df_1d), dtype=bool)
    price_highs = np.zeros(len(df_1d), dtype=bool)
    cci_lows = np.zeros(len(df_1d), dtype=bool)
    cci_highs = np.zeros(len(df_1d), dtype=bool)
    
    for i in range(2, len(df_1d) - 2):
        # Price low: lower than neighbors
        if (df_1d['low'].iloc[i] < df_1d['low'].iloc[i-1] and 
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i-2] and
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i+1] and
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i+2]):
            price_lows[i] = True
        # Price high: higher than neighbors
        if (df_1d['high'].iloc[i] > df_1d['high'].iloc[i-1] and 
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i-2] and
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i+1] and
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i+2]):
            price_highs[i] = True
        # CCI low: lower than neighbors
        if (cci.iloc[i] < cci.iloc[i-1] and 
            cci.iloc[i] < cci.iloc[i-2] and
            cci.iloc[i] < cci.iloc[i+1] and
            cci.iloc[i] < cci.iloc[i+2]):
            cci_lows[i] = True
        # CCI high: higher than neighbors
        if (cci.iloc[i] > cci.iloc[i-1] and 
            cci.iloc[i] > cci.iloc[i-2] and
            cci.iloc[i] > cci.iloc[i+1] and
            cci.iloc[i] > cci.iloc[i+2]):
            cci_highs[i] = True
    
    # Initialize divergence signals
    bullish_div = np.zeros(len(df_1d), dtype=bool)
    bearish_div = np.zeros(len(df_1d), dtype=bool)
    
    # Track recent swing points for divergence detection
    last_price_low = -1
    last_price_high = -1
    last_cci_low = -1
    last_cci_high = -1
    
    for i in range(len(df_1d)):
        if price_lows[i]:
            if last_price_low != -1:
                # Check if current price low is lower than previous price low
                if df_1d['low'].iloc[i] < df_1d['low'].iloc[last_price_low]:
                    # Check if CCI made a higher low
                    if last_cci_low != -1 and cci.iloc[i] > cci.iloc[last_cci_low]:
                        bullish_div[i] = True
            last_price_low = i
            if cci_lows[i]:
                last_cci_low = i
        
        if price_highs[i]:
            if last_price_high != -1:
                # Check if current price high is higher than previous price high
                if df_1d['high'].iloc[i] > df_1d['high'].iloc[last_price_high]:
                    # Check if CCI made a lower high
                    if last_cci_high != -1 and cci.iloc[i] < cci.iloc[last_cci_high]:
                        bearish_div[i] = True
            last_price_high = i
            if cci_highs[i]:
                last_cci_high = i
    
    # Align divergence signals to 4h timeframe
    bullish_div_aligned = align_htf_to_ltf(prices, df_1d, bullish_div.astype(float))
    bearish_div_aligned = align_htf_to_ltf(prices, df_1d, bearish_div.astype(float))
    
    # Trend filter: price relative to 200-period SMA on 4h
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if np.isnan(sma_200[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish divergence on daily + price above 200 SMA (bullish context) + volume confirmation
            if (bullish_div_aligned[i] > 0.5 and 
                close[i] > sma_200[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence on daily + price below 200 SMA (bearish context) + volume confirmation
            elif (bearish_div_aligned[i] > 0.5 and 
                  close[i] < sma_200[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence appears OR price drops below 200 SMA
            if (bearish_div_aligned[i] > 0.5) or (close[i] < sma_200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence appears OR price rises above 200 SMA
            if (bullish_div_aligned[i] > 0.5) or (close[i] > sma_200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals