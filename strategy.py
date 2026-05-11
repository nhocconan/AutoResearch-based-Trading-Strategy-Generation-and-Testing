#!/usr/bin/env python3
# 6h_HTF_LiquidityZone_Reversal
# Hypothesis: Uses 12h liquidity zones (prior swing highs/lows) as support/resistance on 6h timeframe.
# Long when price approaches 12h swing low, shows bullish rejection (close > open), and volume confirms.
# Short when price approaches 12h swing high, shows bearish rejection (close < open), and volume confirms.
# Exit when price reaches opposite liquidity zone or shows rejection at current zone.
# Works in bull markets by buying dips to 12h support and in bear markets by selling rallies to 12h resistance.
# Liquidity zones act as institutional order flow areas where price often reacts.

name = "6h_HTF_LiquidityZone_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for swing points (liquidity zones)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 3:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h swing highs and lows (liquidity zones) ---
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Swing high: higher high followed by lower high
    swing_high = (high_12h[2:] > high_12h[1:-1]) & (high_12h[2:] > high_12h[3:]) & \
                 (high_12h[1:-1] > high_12h[0:-2])
    # Swing low: lower low followed by higher low
    swing_low = (low_12h[2:] < low_12h[1:-1]) & (low_12h[2:] < low_12h[3:]) & \
                (low_12h[1:-1] < low_12h[0:-2])
    
    # Create arrays of swing levels (NaN where no swing)
    swing_high_levels = np.full_like(high_12h, np.nan)
    swing_low_levels = np.full_like(low_12h, np.nan)
    swing_high_levels[2:-1] = high_12h[2:-1][swing_high]
    swing_low_levels[2:-1] = low_12h[2:-1][swing_low]
    
    # Forward fill to maintain levels until next swing
    # Convert to pandas Series for ffill, then back to numpy
    swing_high_series = pd.Series(swing_high_levels)
    swing_low_series = pd.Series(swing_low_levels)
    swing_high_ffill = swing_high_series.ffill().values
    swing_low_ffill = swing_low_series.ffill().values
    
    # --- 6h rejection candles ---
    # Bullish rejection: close > open (bullish body)
    bullish_rej = close > open_
    # Bearish rejection: close < open (bearish body)
    bearish_rej = close < open_
    
    # --- Volume confirmation ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 12h liquidity zones to 6h timeframe
    resistance_zones = align_htf_to_ltf(prices, df_12h, swing_high_ffill)
    support_zones = align_htf_to_ltf(prices, df_12h, swing_low_ffill)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for swing detection (need 3 periods) and volume MA(20)
    start_idx = max(20, 30)  # volume MA needs 20, extra buffer for safety
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(resistance_zones[i]) or
            np.isnan(support_zones[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price levels
        resistance = resistance_zones[i]
        support = support_zones[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        # Distance to zones (as percentage of price)
        dist_to_resistance = abs(close[i] - resistance) / close[i] if resistance > 0 else 1.0
        dist_to_support = abs(close[i] - support) / close[i] if support > 0 else 1.0
        
        # Consider price near zone if within 0.5%
        near_resistance = dist_to_resistance < 0.005
        near_support = dist_to_support < 0.005
        
        if position == 0:
            # Look for rejection near liquidity zones
            if near_support and bullish_rej and vol_spike:
                # Long: price near 12h support, bullish rejection, volume confirmation
                signals[i] = 0.25
                position = 1
            elif near_resistance and bearish_rej and vol_spike:
                # Short: price near 12h resistance, bearish rejection, volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price reaches resistance or shows bearish rejection at support
                if near_resistance or bearish_rej:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches support or shows bullish rejection at resistance
                if near_support or bullish_rej:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals