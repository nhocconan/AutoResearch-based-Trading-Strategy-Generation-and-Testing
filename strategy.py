#!/usr/bin/env python3
"""
6h_market_regime_camellia_1w_v1
Hypothesis: Uses weekly Camarilla pivot levels (from previous week) with a market regime filter based on weekly price position relative to weekly VWAP. In trending regimes (price above/below weekly VWAP), we trade breakouts of weekly R4/S4 levels. In ranging regimes (price near weekly VWAP), we fade at weekly R3/S3 levels. This adapts to both bull and bear markets by following the weekly trend filter. Volume confirmation ensures institutional participation. Designed for 6h timeframe to capture multi-day moves with low frequency (target: 15-35 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_market_regime_camellia_1w_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivots and regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP for regime detection
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_prev = vwap.shift(1).values  # Use previous week's VWAP to avoid look-ahead
    
    # Calculate weekly Camarilla levels (based on previous week)
    weekly_close = df_1w['close'].shift(1).values
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    weekly_range = weekly_high - weekly_low
    
    # Camarilla levels
    r4 = weekly_close + weekly_range * 1.1 / 2
    r3 = weekly_close + weekly_range * 1.1 / 4
    r2 = weekly_close + weekly_range * 1.1 / 6
    r1 = weekly_close + weekly_range * 1.1 / 12
    s1 = weekly_close - weekly_range * 1.1 / 12
    s2 = weekly_close - weekly_range * 1.1 / 6
    s3 = weekly_close - weekly_range * 1.1 / 4
    s4 = weekly_close - weekly_range * 1.1 / 2
    
    # Align all weekly data to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_prev)
    
    # Volume confirmation (24-period average = 4 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vwap_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Regime detection: price relative to weekly VWAP
        price_vs_vwap = close[i] - vwap_aligned[i]
        vwap_threshold = 0.002 * vwap_aligned[i]  # 0.2% threshold for ranging
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 or regime shifts to ranging and price at R3
            if close[i] <= r3_aligned[i] or (abs(price_vs_vwap) < vwap_threshold and close[i] <= r3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above S3 or regime shifts to ranging and price at S3
            if close[i] >= s3_aligned[i] or (abs(price_vs_vwap) < vwap_threshold and close[i] >= s3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trending regime: price significantly above/below weekly VWAP
            if abs(price_vs_vwap) >= vwap_threshold:
                # Strong uptrend: buy breakout of R4 with volume
                if price_vs_vwap > 0 and close[i] >= r4_aligned[i] and vol_confirm:
                    position = 1
                    signals[i] = 0.25
                # Strong downtrend: sell breakdown of S4 with volume
                elif price_vs_vwap < 0 and close[i] <= s4_aligned[i] and vol_confirm:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime: price near weekly VWAP
                # Fade at R3/S3 with volume confirmation
                if close[i] >= r3_aligned[i] and vol_confirm:
                    position = -1
                    signals[i] = -0.25
                elif close[i] <= s3_aligned[i] and vol_confirm:
                    position = 1
                    signals[i] = 0.25
    
    return signals