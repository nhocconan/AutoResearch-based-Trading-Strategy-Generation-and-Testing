# 1D_TRIX_Trend_Volume_Spike
# Hypothesis: Use TRIX on daily timeframe for trend, with weekly trend filter, volume spike, and price near TRIX signal line for entries.
# Long when: TRIX crosses above signal line on daily, weekly trend up, volume spike, price near TRIX.
# Short when: TRIX crosses below signal line on daily, weekly trend down, volume spike, price near TRIX.
# Works in bull/bear by following daily trend and using volume to confirm institutional interest.
# Target: 20-40 trades/year per symbol.

name = "1D_TRIX_Trend_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily TRIX (15-period EMA triple smoothed)
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change() * 100)  # percent change of triple EMA
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_hist = trix - trix_signal
    
    # TRIX crossover signals
    trix_cross_up = (trix_hist > 0) & (trix_hist.shift(1) <= 0)
    trix_cross_down = (trix_hist < 0) & (trix_hist.shift(1) >= 0)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume spike (2x 20-day average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix.iloc[i]) if hasattr(trix, 'iloc') else np.isnan(trix[i]) or
            np.isnan(trix_signal.iloc[i]) if hasattr(trix_signal, 'iloc') else np.isnan(trix_signal[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get scalar values
        trix_val = trix.iloc[i] if hasattr(trix, 'iloc') else trix[i]
        trix_signal_val = trix_signal.iloc[i] if hasattr(trix_signal, 'iloc') else trix_signal[i]
        trix_hist_val = trix_hist.iloc[i] if hasattr(trix_hist, 'iloc') else trix_hist[i]
        trix_hist_prev = trix_hist.iloc[i-1] if hasattr(trix_hist, 'iloc') else trix_hist[i-1]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: TRIX bullish cross + weekly uptrend + volume spike + price near TRIX
            if trix_hist_val > 0 and trix_hist_prev <= 0 and weekly_up and volume_spike:
                # Price near TRIX signal line (within 1%)
                price_near_signal = abs(close[i] - trix_signal_val) / trix_signal_val < 0.01 if trix_signal_val != 0 else False
                if price_near_signal:
                    signals[i] = 0.25
                    position = 1
            # Enter short: TRIX bearish cross + weekly downtrend + volume spike + price near TRIX
            elif trix_hist_val < 0 and trix_hist_prev >= 0 and weekly_down and volume_spike:
                # Price near TRIX signal line (within 1%)
                price_near_signal = abs(close[i] - trix_signal_val) / trix_signal_val < 0.01 if trix_signal_val != 0 else False
                if price_near_signal:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: TRIX turns bearish or weekly trend changes
            if trix_hist_val < 0 or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: TRIX turns bullish or weekly trend changes
            if trix_hist_val > 0 or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals