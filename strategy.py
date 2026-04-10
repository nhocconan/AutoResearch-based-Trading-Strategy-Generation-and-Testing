#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above H3 (bullish Camarilla level) AND 1w EMA(21) > EMA(55) (bullish trend) AND 1d volume > 1.8x 20-bar avg
# - Short when price breaks below L3 (bearish Camarilla level) AND 1w EMA(21) < EMA(55) (bearish trend) AND 1d volume > 1.8x 20-bar avg
# - Exit when price returns to Pivot Point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivot levels provide high-probability reversal zones derived from prior day's range
# - 1w EMA filter ensures alignment with weekly trend to avoid counter-trend trades
# - Volume confirmation filters low-conviction breakouts
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - Works in both bull and bear markets: breakouts capture momentum, pivot reversion captures pullbacks

name = "1d_1w_camarilla_pivot_breakout_volume_trend_v1"
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
    
    # Pre-compute 1d volume confirmation: > 1.8x 20-period average
    volume_1d = prices['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_20_avg_1d)
    
    # Align HTF indicators to 1d timeframe
    ema_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish_1w)
    ema_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish_1w)
    
    # Pre-compute 1d Camarilla pivot levels from prior day's OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Shift by 1 to use prior day's data (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no prior day
    
    # Calculate pivot point and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4)  # Resistance level 3
    l3 = pivot - (range_hl * 1.1 / 4)  # Support level 3
    h4 = pivot + (range_hl * 1.1 / 2)  # Resistance level 4 (stoploss reference)
    l4 = pivot - (range_hl * 1.1 / 2)  # Support level 4 (stoploss reference)
    
    # Breakout conditions
    breakout_long = (close > h3) & ~np.isnan(h3)  # Price closes above H3
    breakout_short = (close < l3) & ~np.isnan(l3)  # Price closes below L3
    
    # Exit conditions: price returns to pivot point (within 0.1% of pivot)
    exit_long = (close <= pivot * 1.001) & ~np.isnan(pivot)
    exit_short = (close >= pivot * 0.999) & ~np.isnan(pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for volume avg
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1w_aligned[i]) or np.isnan(ema_bearish_1w_aligned[i]) or
            np.isnan(vol_spike_1d[i]) or np.isnan(breakout_long[i]) or
            np.isnan(breakout_short[i]) or np.isnan(exit_long[i]) or
            np.isnan(exit_short[i])):
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
                vol_spike_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 1w bearish trend AND volume spike
            elif (breakout_short[i] and 
                  ema_bearish_1w_aligned[i] and 
                  vol_spike_1d[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point
            # Exit when price returns to pivot (mean reversion)
            if position == 1 and exit_long[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short[i]:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals