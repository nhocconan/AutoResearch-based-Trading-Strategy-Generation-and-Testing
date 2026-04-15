#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-based mean reversion with 4h trend filter and volume confirmation
# Long when: price < BB lower(20,2) AND 4h close > 4h EMA50 AND volume > 1.5x 20-period avg AND session 08-20 UTC
# Short when: price > BB upper(20,2) AND 4h close < 4h EMA50 AND volume > 1.5x 20-period avg AND session 08-20 UTC
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 trades over 4 years.
# Mean reversion works in ranging markets; 4h EMA50 filter ensures we only trade counter-trend in the context of higher timeframe trend.
# Volume confirmation avoids false signals in low liquidity periods.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: EMA50 ===
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # === 1h Indicators: Bollinger Bands (20,2) ===
    bb_window = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    bb_upper = ma + (bb_std_dev * bb_std)
    bb_lower = ma - (bb_std_dev * bb_std)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 50) + 5  # BB(20) + EMA50(50) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema_4h_50_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price below BB lower (20,2)
        # 2. 4h close > 4h EMA50 (uptrend on higher timeframe)
        # 3. Volume confirmation
        if (close[i] < bb_lower[i]) and \
           (close[i] > ema_4h_50_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price above BB upper (20,2)
        # 2. 4h close < 4h EMA50 (downtrend on higher timeframe)
        # 3. Volume confirmation
        elif (close[i] > bb_upper[i]) and \
             (close[i] < ema_4h_50_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_BB20_2_4hEMA50_Volume_Filter_Session"
timeframe = "1h"
leverage = 1.0