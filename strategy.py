#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly EMA trend filter with daily price action and volume confirmation
# Weekly EMA34 provides strong trend filter to avoid counter-trend trades
# Daily price above/below weekly EMA with volume confirmation captures momentum
# Low volatility filter avoids whipsaws in choppy markets
# Designed to work in both bull (trend following) and bear (avoiding false signals) markets
# Target: 20-60 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # Daily volume filter: volume > 1.5x 20-period average (moderate to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Volatility filter: ATR below its 30-period median (avoid high volatility chop)
    atr_median = pd.Series(atr14_1w_aligned).rolling(window=30, min_periods=15).median().values
    vol_filter = atr14_1w_aligned < atr_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr14_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly EMA34 + volume filter + low volatility
            if (close[i] > ema34_1w_aligned[i] and 
                volume_filter[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34 + volume filter + low volatility
            elif (close[i] < ema34_1w_aligned[i] and 
                  volume_filter[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below weekly EMA34 (trend change)
            if close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly EMA34 (trend change)
            if close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_weekly_EMA34_vol_filter_v1"
timeframe = "1d"
leverage = 1.0