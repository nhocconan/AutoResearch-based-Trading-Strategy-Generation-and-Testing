#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above H3 (Camarilla resistance) AND 1d EMA(50) > EMA(200) (bullish trend) AND 12h volume > 2.0x 20-bar avg
# - Short when price breaks below L3 (Camarilla support) AND 1d EMA(50) < EMA(200) (bearish trend) AND 12h volume > 2.0x 20-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivots identify key intraday support/resistance levels; 1d EMA filter ensures alignment with daily trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakout in trends, mean reversion in ranges

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1d EMA trend to 12h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Pre-compute 12h Camarilla pivot levels (based on previous 12h bar)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate pivot point (PP) and Camarilla levels using previous bar's OHLC
    # We use rolling window of 2 to get previous bar's values, then shift by 1 to avoid look-ahead
    prev_high = pd.Series(high_12h).shift(1).values
    prev_low = pd.Series(low_12h).shift(1).values
    prev_close = pd.Series(close_12h).shift(1).values
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla levels
    range_hl = prev_high - prev_low
    h3 = pp + (range_hl * 1.1 / 4.0)  # Resistance level 3
    l3 = pp - (range_hl * 1.1 / 4.0)  # Support level 3
    h4 = pp + (range_hl * 1.1 / 2.0)  # Resistance level 4 (stronger breakout)
    l4 = pp - (range_hl * 1.1 / 2.0)  # Support level 4 (stronger breakout)
    
    # Breakout conditions: price breaks above H3 or below L3 (using current bar's close)
    breakout_long = close_12h > h3
    breakout_short = close_12h < l3
    
    # Exit condition: price returns to pivot point (mean reversion)
    exit_long = close_12h < pp  # Exit long when price falls below pivot
    exit_short = close_12h > pp  # Exit short when price rises above pivot
    
    # Pre-compute 12h volume confirmation: > 2.0x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid (NaN from rolling/shift)
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(breakout_long[i]) or np.isnan(breakout_short[i]) or
            np.isnan(exit_long[i]) or np.isnan(exit_short[i]) or
            np.isnan(vol_spike[i]) or np.isnan(pp[i]) or np.isnan(h3[i]) or np.isnan(l3[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 1d bullish trend AND volume spike
            if (breakout_long[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 1d bearish trend AND volume spike
            elif (breakout_short[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit when price returns to pivot point
            if position == 1:  # Long position
                if exit_long[i]:  # Price fell below pivot
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Stay long
            else:  # Short position
                if exit_short[i]:  # Price rose above pivot
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Stay short
    
    return signals