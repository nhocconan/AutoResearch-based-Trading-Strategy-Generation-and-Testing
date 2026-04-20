#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action relative to 12h VWAP with volume confirmation and ATR stop
# - Uses 12h VWAP as dynamic equilibrium: price > VWAP suggests bullish bias, < VWAP bearish
# - Entry: price crosses 12h VWAP with volume > 2x 20-period average (strong conviction)
# - Exit: price reverts back across 12h VWAP or ATR stop hit (1.5x ATR for tight risk)
# - VWAP provides mean-reversion edge in ranging markets while capturing trends
# - High volume filter ensures only significant breaks trigger entries
# - Target: 25-40 trades per year per symbol (100-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    pv_12h = typical_price_12h * volume_12h
    
    # Calculate cumulative VWAP (session-based, reset daily)
    # Since we don't have session boundaries, use rolling window as approximation
    # Use 28 periods (2 * 14) to approximate daily VWAP on 12h timeframe
    cum_pv = np.nancumsum(pv_12h)
    cum_vol = np.nancumsum(volume_12h)
    vwap_12h = np.where(cum_vol != 0, cum_pv / cum_vol, typical_price_12h)
    
    # Alternative: rolling VWAP approximation for stability
    # Use 28-period rolling window (~1 day on 12h chart)
    window = 28
    pv_sum = pd.Series(pv_12h).rolling(window=window, min_periods=window).sum().values
    vol_sum = pd.Series(volume_12h).rolling(window=window, min_periods=window).sum().values
    vwap_12h_roll = np.where(vol_sum != 0, pv_sum / vol_sum, typical_price_12h)
    
    # Use rolling VWAP as primary (more stable)
    vwap_12h = vwap_12h_roll
    
    # Align VWAP to 4h timeframe
    vwap_12h_4h = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Calculate ATR for stop loss (using 12h data)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_4h = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(vwap_12h_4h[i]) or np.isnan(atr_12h_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vwap = vwap_12h_4h[i]
        
        if position == 0:
            # Long entry: price crosses above VWAP with volume surge
            if price > vwap and price <= vwap * 1.001 and vol > 2.0 * vol_ma[i]:  # crossed above
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price crosses below VWAP with volume surge
            elif price < vwap and price >= vwap * 0.999 and vol > 2.0 * vol_ma[i]:  # crossed below
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses back below VWAP OR ATR stop hit (1.5*ATR)
            if price < vwap or price < entry_price - 1.5 * atr_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above VWAP OR ATR stop hit (1.5*ATR)
            if price > vwap or price > entry_price + 1.5 * atr_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Cross_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0