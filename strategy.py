#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R (14) mean reversion with 1w EMA trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) + price > 1w EMA34 (uptrend) + volume > 1.3x 20-period avg
# Short when Williams %R > -20 (overbought) + price < 1w EMA34 (downtrend) + volume > 1.3x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag
# Williams %R identifies exhaustion points; 1w EMA ensures we trade with the higher timeframe trend
# Volume confirmation reduces false signals in low-participation moves

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Indicator: Williams %R (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = williams_r.values  # convert to numpy array
    
    # Align Williams %R to 1d timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w HTF data once before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # === 1w Indicator: EMA (34-period) for trend direction ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 1d timeframe (wait for 1w bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need 1d data for Williams %R (14) + 1w data for EMA34 (34) + volume(20) + buffer
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold condition)
        # 2. Price above 1w EMA34 (uptrend on higher timeframe)
        # 3. Volume confirmation
        if (williams_r_aligned[i] < -80) and \
           (close[i] > ema_34_1w_aligned[i]) and \
           vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought condition)
        # 2. Price below 1w EMA34 (downtrend on higher timeframe)
        # 3. Volume confirmation
        elif (williams_r_aligned[i] > -20) and \
             (close[i] < ema_34_1w_aligned[i]) and \
             vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_WilliamsR14_1wEMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0