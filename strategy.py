#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR(14) volatility filter.
# Enter long when price breaks above 20-period Donchian high, 1d EMA34 trending up, and ATR(14) > 0.5 * ATR(50) (ensuring sufficient volatility).
# Enter short when price breaks below 20-period Donchian low, 1d EMA34 trending down, and ATR(14) > 0.5 * ATR(50).
# Exit when price crosses the 1d EMA34 or reaches the opposite Donchian level.
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining exposure.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid excessive fee churn.
# Donchian channels provide clear breakout levels; 1d EMA34 filters for higher-timeframe trend alignment;
# ATR ratio filter ensures trades occur only during sufficient volatility, reducing whipsaws in low-volatility regimes.

name = "4h_DonchianBreakout_1dEMA34_ATRVolFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    vol_filter = atr_ratio > 0.5  # Ensure sufficient volatility
    
    # Calculate Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter
        vol_ok = vol_filter[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high[i]
        breakout_down = close[i] < donch_low[i]
        
        # Exit conditions: price crosses 1d EMA34 or reaches opposite Donchian level
        exit_long = close[i] < ema_34_aligned[i] or close[i] < donch_low[i]
        exit_short = close[i] > ema_34_aligned[i] or close[i] > donch_high[i]
        
        # Handle entries and exits
        if breakout_up and ema_trend_up and vol_ok and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and ema_trend_down and vol_ok and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals