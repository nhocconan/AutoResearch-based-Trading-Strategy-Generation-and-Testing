#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# Long when price breaks above Donchian upper band AND price > weekly EMA20 AND volume > 1.5x average
# Short when price breaks below Donchian lower band AND price < weekly EMA20 AND volume > 1.5x average
# Exit when price crosses back through Donchian midline (10-period) or trailing stop hits 2*ATR
# Uses daily timeframe to reduce trade frequency, targets 30-100 total trades over 4 years
# Works in bull markets via trend-following breakouts; works in bear markets via short breakdowns
# Weekly EMA filter ensures we only trade with the higher timeframe trend

name = "1d_donchian_1w_ema_vol_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i]) or \
           np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit conditions: price crosses below Donchian midline OR trailing stop hit
            if close[i] < donchian_mid[i] or close[i] < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions: price crosses above Donchian midline OR trailing stop hit
            if close[i] > donchian_mid[i] or close[i] > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation and weekly trend filter
            # Long breakout: price above Donchian upper band AND price > weekly EMA20
            if (close[i] > donchian_upper[i] and close[i] > ema_20_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                atr_at_entry = atr[i]
            # Short breakdown: price below Donchian lower band AND price < weekly EMA20
            elif (close[i] < donchian_lower[i] and close[i] < ema_20_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                atr_at_entry = atr[i]
    
    return signals