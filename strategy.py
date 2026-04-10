#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above H3 (bullish bias) AND 1w EMA(21) > EMA(55) (bullish trend) AND 1d volume > 1.5x 20-bar avg
# - Short when price breaks below L3 (bearish bias) AND 1w EMA(21) < EMA(55) (bearish trend) AND 1d volume > 1.5x 20-bar avg
# - Exit when price returns to H4/L4 levels (pivot reversion)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla levels from 1d capture intraday support/resistance
# - 1w EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, pivot reversion in ranges

name = "12h_1w_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 55 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(21) vs EMA(55)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_55_1w = pd.Series(close_1w).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_bullish_1w = ema_21_1w > ema_55_1w
    ema_bearish_1w = ema_21_1w < ema_55_1w
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # L2 = close - 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    
    rang = high_1d - low_1d
    h4 = close_1d + 1.1 * rang / 2
    l4 = close_1d - 1.1 * rang / 2
    h3 = close_1d + 1.1 * rang / 4
    l3 = close_1d - 1.1 * rang / 4
    
    # Align HTF indicators to 12h timeframe
    ema_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish_1w)
    ema_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish_1w)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Current price
    close = prices['close'].values
    
    # Breakout conditions
    breakout_long = close > h3_aligned  # Price above H3 (bullish bias)
    breakout_short = close < l3_aligned  # Price below L3 (bearish bias)
    
    # Reversion conditions (exit when price returns to H4/L4)
    revert_long = close < h4_aligned  # Price below H4 (exit long)
    revert_short = close > l4_aligned  # Price above L4 (exit short)
    
    # Session filter: 00-23 UTC (12h timeframe, less restrictive)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 12h, but kept for consistency
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1w_aligned[i]) or np.isnan(ema_bearish_1w_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter (always true for 12h, but kept for structure)
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 1w bullish trend AND volume spike
            if (breakout_long[i] and 
                ema_bullish_1w_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 1w bearish trend AND volume spike
            elif (breakout_short[i] and 
                  ema_bearish_1w_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for reversion to H4/L4
            # Exit when price returns to H4/L4 levels (pivot reversion)
            if position == 1:  # Long position
                exit_signal = revert_long[i]  # Price below H4
            else:  # Short position
                exit_signal = revert_short[i]  # Price above L4
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals