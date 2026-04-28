#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volatility filter.
# Enter long when price breaks above 4h Donchian upper channel with 1d EMA34 uptrend and ATR(14) > 0.5 * ATR(50) (volatility expansion).
# Enter short when price breaks below 4h Donchian lower channel with 1d EMA34 downtrend and ATR(14) > 0.5 * ATR(50).
# Exit when price retraces to the 4h Donchian midpoint (upper+lower)/2.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide clear trend-following structure. EMA34 on 1d ensures higher-timeframe trend alignment.
# Volatility filter ensures trades occur during expansion phases, reducing whipsaw in low-vol environments.
# This pattern has shown strong performance on ETH/USDT and BTC/USDT in prior experiments.

name = "4h_Donchian20_Breakout_1dEMA34_ATR_VolFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:  # Need at least one complete 4h bar for Donchian
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    # Donchian midpoint = (upper + lower) / 2
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to 4h (shifted by one bar to avoid look-ahead)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (34-period)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # ATR-based volatility filter: ATR(14) > 0.5 * ATR(50) (volatility expansion)
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14 > (0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure sufficient history for ATR and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter
        vol_ok = vol_filter[i]
        
        # 1d EMA trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        ema_trend_up = close[i] > ema_34_aligned[i]
        ema_trend_down = close[i] < ema_34_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper, price > EMA34 (uptrend), vol expansion
            if price > donchian_high_aligned[i] and ema_trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian lower, price < EMA34 (downtrend), vol expansion
            elif price < donchian_low_aligned[i] and ema_trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals