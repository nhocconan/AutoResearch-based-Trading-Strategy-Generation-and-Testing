#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Camarilla pivot levels from 1d: breakout above H4 = long, below L4 = short
# - 1w EMA(21) trend filter: price above EMA = bullish bias, below = bearish bias
# - Volume confirmation: current 1d volume > 2.0x 20-period average
# - ATR-based trailing stop: exit long when price < highest_high - 2.0*ATR, exit short when price > lowest_low + 2.0*ATR
# - Designed for 1d timeframe: targets 15-30 trades/year to avoid fee drag
# - Works in bull/bear markets: 1w EMA filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "1d_1w_camarilla_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h4 = pp + (range_1d * 1.1 / 2)  # H4 = PP + 1.1 * (High - Low) / 2
    l4 = pp - (range_1d * 1.1 / 2)  # L4 = PP - 1.1 * (High - Low) / 2
    h3 = pp + (range_1d * 1.1 / 4)  # H3 = PP + 1.1 * (High - Low) / 4
    l3 = pp - (range_1d * 1.1 / 4)  # L3 = PP - 1.1 * (High - Low) / 4
    
    # Pre-compute 1d volume confirmation
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * avg_volume_20)
    
    # Pre-compute 1d ATR(14) for trailing stop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0  # for trailing stop
    lowest_low = 0.0    # for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            if close_1d[i] > highest_high:
                highest_high = close_1d[i]
            # Exit: trailing stop hit OR price re-enters Camarilla H3-L3 range (failed breakout)
            if close_1d[i] < highest_high - 2.0 * atr_14[i] or (close_1d[i] < h3[i] and close_1d[i] > l3[i]):
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            if close_1d[i] < lowest_low:
                lowest_low = close_1d[i]
            # Exit: trailing stop hit OR price re-enters Camarilla H3-L3 range (failed breakout)
            if close_1d[i] > lowest_low + 2.0 * atr_14[i] or (close_1d[i] > l3[i] and close_1d[i] < h3[i]):
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i]:
                # Determine trend bias from 1w EMA
                bullish_bias = close_1d[i] > ema_21_1w_aligned[i]
                bearish_bias = close_1d[i] < ema_21_1w_aligned[i]
                
                # Breakout long: price closes above H4 with bullish bias
                if bullish_bias and close_1d[i] > h4[i]:
                    position = 1
                    entry_price = close_1d[i]
                    highest_high = close_1d[i]
                    signals[i] = 0.25
                # Breakout short: price closes below L4 with bearish bias
                elif bearish_bias and close_1d[i] < l4[i]:
                    position = -1
                    entry_price = close_1d[i]
                    lowest_low = close_1d[i]
                    signals[i] = -0.25
    
    return signals