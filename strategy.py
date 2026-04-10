#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + VWAP + 1w trend filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
# - VWAP acts as dynamic support/resistance; price > VWAP = bullish bias, < VWAP = bearish bias
# - 1w EMA50 trend filter ensures we only trade with the weekly trend to avoid counter-trend whipsaws
# - Long when Bull Power > 0, price > VWAP, and 1w close > 1w EMA50
# - Short when Bear Power > 0, price < VWAP, and 1w close < 1w EMA50
# - Exit when Elder Power reverses (Bull Power < 0 for longs, Bear Power < 0 for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_1w_elder_ray_vwap_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Elder Ray components (13-period EMA)
    close_s = pd.Series(prices['close'])
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'] - ema13  # High - EMA13
    bear_power = ema13 - prices['low']   # EMA13 - Low
    
    # Pre-compute VWAP (typical price * volume cumulative)
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3.0
    vwap_numerator = (typical_price * prices['volume']).cumsum()
    vwap_denominator = prices['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    # Handle initial NaN from zero volume
    vwap = np.where(vwap_denominator == 0, np.nan, vwap)
    
    # Pre-compute aligned 1w EMA(50) for trend filter
    close_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vwap[i]) or np.isnan(close_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power positive, price above VWAP, weekly uptrend
            if (bull_power[i] > 0 and 
                prices['close'].iloc[i] > vwap[i] and 
                prices['close'].iloc[i] > close_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power positive, price below VWAP, weekly downtrend
            elif (bear_power[i] > 0 and 
                  prices['close'].iloc[i] < vwap[i] and 
                  prices['close'].iloc[i] < close_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when Elder Power reverses (loss of momentum)
            if position == 1:  # Long position
                if bull_power[i] < 0:  # Bull Power turned negative
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if bear_power[i] < 0:  # Bear Power turned negative
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals