#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ATR filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 1h
# - Volume filter: 4h volume > 1.3x 20-period average volume (institutional participation)
# - Volatility filter: 1d ATR(14) < 0.05 * price (avoid extreme volatility)
# - Session filter: 08-20 UTC (active London/NY overlap)
# - Position size: 0.20 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 1h
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Breakouts capture strong moves; filters avoid chop/false signals

name = "1h_4h_1d_camarilla_volume_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h volume spike filter
    volume_4h = df_4h['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    # Pre-compute 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_filter = (atr_14 / close_1d) < 0.05  # ATR < 5% of price
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Pre-compute 1h Camarilla levels (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    # Since we have 1h data, we'll approximate using rolling window of 24 periods (1 day)
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    open_1h = prices['open'].values
    
    # Approximate daily OHLC using 24-period rolling (for 1h timeframe)
    # This is an approximation but avoids look-ahead by using completed periods
    def rolling_ohlc(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).apply(
            lambda x: x[-1] if len(x) == window else np.nan, raw=True
        ).values
    
    # Get previous day's close (24 periods ago) for Camarilla calculation
    prev_close = np.roll(close_1h, 24)
    prev_high = np.roll(high_1h, 24)
    prev_low = np.roll(low_1h, 24)
    
    # Camarilla levels: H3/L3 = close +- (high-low)*1.1/4
    camarilla_high = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_low = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Pre-compute 1h ATR(14) for stoploss
    tr_1h1 = high_1h - low_1h
    tr_1h2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr_1h3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr_1h = np.maximum(tr_1h1, np.maximum(tr_1h2, tr_1h3))
    tr_1h[0] = tr_1h1[0]
    atr_14_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_filter_aligned[i]) or
            np.isnan(atr_14_1h[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price re-enters Camarilla range OR stoploss hit
            if (close_1h[i] < camarilla_high[i] and close_1h[i] > camarilla_low[i]) or \
               close_1h[i] < entry_price - 1.5 * atr_14_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price re-enters Camarilla range OR stoploss hit
            if (close_1h[i] < camarilla_high[i] and close_1h[i] > camarilla_low[i]) or \
               close_1h[i] > entry_price + 1.5 * atr_14_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakouts with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter_aligned[i]:
                # Long: price breaks above Camarilla H3
                if close_1h[i] > camarilla_high[i]:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: price breaks below Camarilla L3
                elif close_1h[i] < camarilla_low[i]:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals