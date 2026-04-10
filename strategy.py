#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d regime filter
# - Long when price breaks above H3 (resistance) AND 4h close > 4h open (bullish) AND 1d volume > 1.3x 20-bar avg
# - Short when price breaks below L3 (support) AND 4h close < 4h open (bearish) AND 1d volume > 1.3x 20-bar avg
# - Exit when price returns to H4/L4 or opposite pivot level
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Camarilla pivots identify key intraday support/resistance levels
# - 4h candle direction ensures alignment with higher timeframe momentum
# - Volume confirmation avoids low-liquidity false breakouts
# - Session filter (08-20 UTC) reduces noise trades
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in both bull and bear markets: breakouts capture momentum, filters prevent whipsaws

name = "1h_4h_1d_camarilla_breakout_volume_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h candle direction: bullish if close > open
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    candle_bullish_4h = close_4h > open_4h
    candle_bearish_4h = close_4h < open_4h
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.3 * volume_20_avg_1d)
    
    # Align HTF indicators to 1h timeframe
    candle_bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, candle_bullish_4h)
    candle_bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, candle_bearish_4h)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Camarilla pivots on 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily range (using prior bar's high/low for intraday calculation)
    # For bar i, use high[i-1] and low[i-1] to avoid look-ahead
    daily_range = np.roll(high, 1) - np.roll(low, 1)
    daily_range[0] = high[0] - low[0]  # First bar uses its own range
    
    # Camarilla levels based on prior bar's close
    prior_close = np.roll(close, 1)
    prior_close[0] = close[0]  # First bar uses its own close
    
    # Resistance levels
    H4 = prior_close + 1.1 * daily_range / 2
    H3 = prior_close + 1.1 * daily_range / 4
    H2 = prior_close + 1.1 * daily_range / 6
    H1 = prior_close + 1.1 * daily_range / 12
    
    # Support levels
    L1 = prior_close - 1.1 * daily_range / 12
    L2 = prior_close - 1.1 * daily_range / 6
    L3 = prior_close - 1.1 * daily_range / 4
    L4 = prior_close - 1.1 * daily_range / 2
    
    # Breakout conditions
    breakout_long = close > H3  # Price breaks above H3 resistance
    breakout_short = close < L3  # Price breaks below L3 support
    
    # Exit conditions: return to H4/L4 or opposite pivot
    exit_long = close >= H4 or close <= L1  # Long exit: hit H4 or reverse to L1
    exit_short = close <= L4 or close >= H1  # Short exit: hit L4 or reverse to H1
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(candle_bullish_4h_aligned[i]) or np.isnan(candle_bearish_4h_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(breakout_long[i]) or
            np.isnan(breakout_short[i]) or np.isnan(exit_long[i]) or np.isnan(exit_short[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h bullish candle AND volume spike
            if (breakout_long[i] and 
                candle_bullish_4h_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h bearish candle AND volume spike
            elif (breakout_short[i] and 
                  candle_bearish_4h_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit based on position type
            if position == 1:  # Long position
                if exit_long[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # Short position
                if exit_short[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals