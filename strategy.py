#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for direction and 1h volume/price action for entry timing.
# Uses 4h Donchian(20) breakouts to establish trend direction, then enters on 1h pullbacks to VWAP
# with volume confirmation. Includes session filter (08-20 UTC) to reduce noise. Designed to work
# in both bull and bear markets by following higher timeframe trend with selective entries.
name = "1h_Donchian20_VWAP_Pullback_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian(20) for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe (wait for 4h bar close)
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1h VWAP for entry timing (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # 1h volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session or missing data
        if not in_session[i] or np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: 4h Donchian breakout up + price pulls back to VWAP with volume
            if (donchian_high_1h[i] > donchian_high_1h[i-1] and  # New 4h high
                price <= vwap[i] * 1.005 and price >= vwap[i] * 0.995 and  # Near VWAP
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h Donchian breakout down + price pulls back to VWAP with volume
            elif (donchian_low_1h[i] < donchian_low_1h[i-1] and  # New 4h low
                  price <= vwap[i] * 1.005 and price >= vwap[i] * 0.995 and  # Near VWAP
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h Donchian breaks down or price moves 1.5% away from VWAP
            if (donchian_low_1h[i] < donchian_low_1h[i-1] or  # 4h Donchian low broken
                price < vwap[i] * 0.985 or price > vwap[i] * 1.015):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h Donchian breaks up or price moves 1.5% away from VWAP
            if (donchian_high_1h[i] > donchian_high_1h[i-1] or  # 4h Donchian high broken
                price < vwap[i] * 0.985 or price > vwap[i] * 1.015):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals