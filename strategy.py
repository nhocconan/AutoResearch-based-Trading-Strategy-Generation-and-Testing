#!/usr/bin/env python3
"""
1h Time-Based Trend Reversal with Volume Filter
Hypothesis: During low volatility overnight sessions (00-08 UTC), price often reverts to the 
4-hour VWAP. During active sessions (08-20 UTC), we follow 4-hour momentum with volume confirmation.
Designed for 15-30 trades/year on 1h timeframe (60-120 total over 4 years).
Works in bull markets by riding 4h uptrends and in bear markets by fading overnight reversions.
"""

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
    
    # Pre-compute hour filter for efficiency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4H data once
    df_4h = get_htf_data(prices, '4h')
    
    # 4H VWAP (volume weighted average price)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap_4h = (typical_price_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h_values = vwap_4h.values
    
    # 4H EMA34 for trend filter
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4H indicators to 1H timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_values)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1H volume spike: 1.5x 24-period average
    vol_ma_1h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma_1h)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or
            np.isnan(vol_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        price = close[i]
        vwap = vwap_4h_aligned[i]
        ema34 = ema34_4h_aligned[i]
        
        # Session filter: 08-20 UTC active, 00-08 UTC overnight
        is_active_session = 8 <= hour <= 20
        
        if position == 0:
            if is_active_session:
                # Active session: follow 4H trend with volume confirmation
                if price > ema34 and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                elif price < ema34 and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
            else:
                # Overnight session: mean revert to 4H VWAP
                if price < vwap * 0.995:  # Oversold threshold
                    signals[i] = 0.20
                    position = 1
                elif price > vwap * 1.005:  # Overbought threshold
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.20
            # Exit conditions
            if is_active_session:
                # In active session: exit on trend reversal
                if price < ema34:
                    signals[i] = 0.0
                    position = 0
            else:
                # In overnight: exit when price returns to VWAP
                if price >= vwap:
                    signals[i] = 0.0
                    position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.20
            # Exit conditions
            if is_active_session:
                # In active session: exit on trend reversal
                if price > ema34:
                    signals[i] = 0.0
                    position = 0
            else:
                # In overnight: exit when price returns to VWAP
                if price <= vwap:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "1h_TimeBased_Trend_Reversion_Volume"
timeframe = "1h"
leverage = 1.0