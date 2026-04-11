#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_vwap_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h VWAP (volume-weighted average price)
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_12h = (typical_price_12h * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h = vwap_12h.values
    
    # Shift by 1 to use only completed 12h bars (prevent look-ahead)
    vwap_12h = np.roll(vwap_12h, 1)
    vwap_12h[0] = np.nan
    
    # Align 12h VWAP to 6h timeframe
    vwap_6h = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # 6h VWAP for short-term mean reversion
    typical_price_6h = (high + low + close) / 3
    vwap_6h_calc = (typical_price_6h * volume).cumsum() / volume.cumsum()
    vwap_6h_calc = np.concatenate([[np.nan], vwap_6h_calc[1:]])  # first value NaN
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_6h[i]) or np.isnan(vwap_6h_calc[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_low = low[i]
        price_high = high[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.2 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.0025 * price_close  # ATR > 0.25% of price
        
        # Distance from 12h VWAP as percentage
        dist_from_vwap = (price_close - vwap_6h[i]) / vwap_6h[i]
        
        # Long: price significantly below 12h VWAP with volume and volatility
        long_signal = volume_confirmed and vol_filter and (dist_from_vwap < -0.015)  # 1.5% below
        
        # Short: price significantly above 12h VWAP with volume and volatility
        short_signal = volume_confirmed and vol_filter and (dist_from_vwap > 0.015)   # 1.5% above
        
        # Exit when price returns to 6h VWAP (short-term mean)
        exit_long = position == 1 and price_close > vwap_6h_calc[i]
        exit_short = position == -1 and price_close < vwap_6h_calc[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h VWAP acts as dynamic support/resistance for 6h price action.
# Enters long when 6h price deviates >1.5% below 12h VWAP with volume confirmation (>1.2x average)
# and sufficient volatility (ATR > 0.25% of price), expecting mean reversion to 12h VWAP.
# Enters short when price deviates >1.5% above 12h VWAP with same conditions.
# Exits when price returns to 6h VWAP (short-term mean), capturing the reversion.
# Works in both bull (buying dips to VWAP support) and bear (selling rallies to VWAP resistance).
# Volume and volatility filters prevent false signals in low-activity periods.