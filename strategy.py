#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d ATR volume filter and 1w Regime filter (ADX)
# - Long: price closes above Camarilla H3, 1d ATR > 1.2x 20-period ATR average, 1w ADX > 20
# - Short: price closes below Camarilla L3, 1d ATR > 1.2x 20-period ATR average, 1w ADX > 20
# - Exit: price returns to Camarilla H4/L4 levels
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Uses 1w ADX regime filter to avoid ranging markets and reduce whipsaw
# - Uses 1d ATR volume confirmation to ensure breakouts have sufficient volatility

name = "12h_1d_1w_camarilla_atr_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1w data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Load 1d data ONCE before loop for Camarilla pivots and ATR volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1d ATR for volume filter (20-period average)
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_sma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_sma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_20_1d)
    
    # Pre-compute 12h ATR for stoploss reference (not used for exit in this version)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or np.isnan(atr_sma_20_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        atr_current = atr_1d[i]  # Current 1d ATR
        
        # Camarilla levels
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        h4_level = h4_aligned[i]
        l4_level = l4_aligned[i]
        
        # ATR volume filter: current 1d ATR > 1.2x 20-period average (ensures volatile breakout)
        atr_filter = atr_current > 1.2 * atr_sma_20_aligned[i]
        
        # Regime filter: 1w ADX > 20 (avoid ranging markets)
        regime_filter = adx_aligned[i] > 20
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above Camarilla H3, ATR filter, trending regime
        if close_price > h3_level and atr_filter and regime_filter:
            enter_long = True
        
        # Short breakout: price closes below Camarilla L3, ATR filter, trending regime
        if close_price < l3_level and atr_filter and regime_filter:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Camarilla H4 level
            exit_long = close_price >= h4_level
        elif position == -1:
            # Exit short if price returns to Camarilla L4 level
            exit_short = close_price <= l4_level
        
        # Track entry price (though not used for stoploss in this version)
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals