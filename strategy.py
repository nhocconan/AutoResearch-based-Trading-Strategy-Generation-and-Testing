# 6h_TRIX_RSI_Confluence_v1
# TRIX (12) for momentum direction, RSI(14) for overbought/oversold, with 1d trend filter and volume confirmation
# TRIX crosses signal line with RSI confirmation and 1d EMA50 trend alignment
# Designed for 6h timeframe with 1d HTF filter to capture medium-term momentum with controlled frequency
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag while maintaining edge
# Works in both bull and bear markets by using momentum reversal logic with trend filter

name = "6h_TRIX_RSI_Confluence_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (1d)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # === TRIX indicator (6h) ===
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * (pd.Series(ema3).pct_change()).values
    
    # TRIX signal line (EMA of TRIX, 9)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_hist = trix_raw - trix_signal  # Histogram for crossover signals
    
    # === RSI(14) (6h) ===
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(trix_hist[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: TRIX bullish crossover + RSI not overbought + above 1d EMA50 + volume spike
            if (trix_hist[i] > 0 and trix_hist[i-1] <= 0 and  # TRIX crosses above signal
                rsi[i] < 70 and 
                close[i] > ema50_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish crossover + RSI not oversold + below 1d EMA50 + volume spike
            elif (trix_hist[i] < 0 and trix_hist[i-1] >= 0 and  # TRIX crosses below signal
                  rsi[i] > 30 and
                  close[i] < ema50_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position == 1:
            # Exit long: TRIX bearish crossover OR RSI overbought OR below 1d EMA50
            if (trix_hist[i] < 0 and trix_hist[i-1] >= 0 or  # TRIX crosses below signal
                rsi[i] > 75 or
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX bullish crossover OR RSI oversold OR above 1d EMA50
            if (trix_hist[i] > 0 and trix_hist[i-1] <= 0 or  # TRIX crosses above signal
                rsi[i] < 25 or
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals