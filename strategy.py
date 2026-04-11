#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d volume confirmation and 1w ADX regime filter
# - Long: bullish fractal break above recent high, volume > 1.5x 20-period avg, 1w ADX(14) > 20
# - Short: bearish fractal break below recent low, volume > 1.5x 20-period avg, 1w ADX(14) > 20
# - Exit: opposite fractal touch or ATR-based stop (2x ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams fractals provide high-probability reversal/breakout points with confirmation
# - 1w ADX > 20 ensures we trade only when weekly trend is present, reducing whipsaw in ranging markets
# - Works in bull (breakouts with trend) and bear (fading false breaks via regime filter)

name = "12h_1d_1w_fractal_adx_volume_v1"
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
    
    # Load 1d data ONCE before loop for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return signals
    
    # Pre-compute 1d Williams fractals (requires 5-bar window: n-2, n-1, n, n+1, n+2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bullish fractal: lowest low in middle with two higher lows on each side
    bullish_fractal = np.zeros(len(high_1d))
    # Bearish fractal: highest high in middle with two lower highs on each side
    bearish_fractal = np.zeros(len(high_1d))
    
    for i in range(2, len(high_1d) - 2):
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
    
    # Align 1d fractals to 12h timeframe with extra delay (fractals need 2 bars to confirm)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute recent 12h high/low for fractal breakout confirmation (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Fractal levels
        bullish_fractal_level = bullish_aligned[i]
        bearish_fractal_level = bearish_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Regime filter: 1w ADX > 20 (indicates non-ranging market)
        adx_regime = adx_aligned[i] > 20
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above bullish fractal AND above recent 20-period high
        if (bullish_fractal_level > 0 and  # valid fractal exists
            close_price > bullish_fractal_level and
            close_price > highest_high_20[i] and  # breakout confirmation
            vol_confirm and adx_regime):
            enter_long = True
        
        # Short breakout: price below bearish fractal AND below recent 20-period low
        if (bearish_fractal_level > 0 and  # valid fractal exists
            close_price < bearish_fractal_level and
            close_price < lowest_low_20[i] and  # breakdown confirmation
            vol_confirm and adx_regime):
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches bearish fractal or ATR-based stop
            exit_long = (bearish_fractal_level > 0 and close_price <= bearish_fractal_level) or \
                        (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price touches bullish fractal or ATR-based stop
            exit_short = (bullish_fractal_level > 0 and close_price >= bullish_fractal_level) or \
                         (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
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