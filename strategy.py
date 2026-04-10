#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when close breaks above H3 pivot level AND 1w EMA(21) > EMA(55) (bullish trend) AND volume > 1.5x 20-bar avg
# - Short when close breaks below L3 pivot level AND 1w EMA(21) < EMA(55) (bearish trend) AND volume > 1.5x 20-bar avg
# - Exit when price returns to Pivot Point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivots identify key intraday support/resistance levels
# - 1w EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Works in both bull and bear markets: breakouts capture trends, mean reversion exit works in ranges

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(21) vs EMA(55)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_55_1w = pd.Series(close_1w).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_bullish_1w = ema_21_1w > ema_55_1w
    ema_bearish_1w = ema_21_1w < ema_55_1w
    
    # Align HTF indicators to 1d timeframe
    ema_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish_1w)
    ema_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish_1w)
    
    # Pre-compute Camarilla pivot levels for 1d (using previous day's OHLC)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Shift to get previous day's OHLC for today's pivot calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous day
    
    # Calculate pivot point and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4)
    l3 = pivot - (range_hl * 1.1 / 4)
    h4 = pivot + (range_hl * 1.1 / 2)
    l4 = pivot - (range_hl * 1.1 / 2)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    # Breakout conditions
    breakout_long = close > h3
    breakout_short = close < l3
    
    # Exit condition: price returns to pivot point (mean reversion)
    exit_long = close < pivot  # Exit long when price falls below pivot
    exit_short = close > pivot  # Exit short when price rises above pivot
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1w_aligned[i]) or np.isnan(ema_bearish_1w_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(breakout_long[i]) or np.isnan(breakout_short[i]) or
            np.isnan(exit_long[i]) or np.isnan(exit_short[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 1w bullish trend AND volume spike
            if (breakout_long[i] and 
                ema_bullish_1w_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 1w bearish trend AND volume spike
            elif (breakout_short[i] and 
                  ema_bearish_1w_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot point
            if position == 1:
                if exit_long[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if exit_short[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals