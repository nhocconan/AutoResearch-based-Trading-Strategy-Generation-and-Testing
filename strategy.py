#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA34 trend filter and 1d volume spike (>2.0x 20-bar average) captures institutional breakouts with trend alignment. Uses discrete sizing (0.20) to target ~20-40 trades/year. Works in bull/bear by only taking breakouts aligned with 4h trend. Volume filter ensures participation. Session filter (08-20 UTC) reduces noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Load 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Load 1d data ONCE before loop for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume MA(20) for volume confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla levels for 1h (based on previous day's OHLC)
    # We need to group by day and calculate pivots from previous day
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    # Shift by 1 to get previous day's data
    df['prev_high'] = df.groupby('date')['high'].shift(1)
    df['prev_low'] = df.groupby('date')['low'].shift(1)
    df['prev_close'] = df.groupby('date')['close'].shift(1)
    # Forward fill within day to get previous day's values for all bars of current day
    df['prev_high'] = df.groupby('date')['prev_high'].ffill()
    df['prev_low'] = df.groupby('date')['prev_low'].ffill()
    df['prev_close'] = df.groupby('date')['prev_close'].ffill()
    
    # Calculate Camarilla levels
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    rng = df['prev_high'] - df['prev_low']
    camarilla_r1 = df['prev_close'] + (rng * 1.1 / 12)
    camarilla_s1 = df['prev_close'] - (rng * 1.1 / 12)
    r1 = camarilla_r1.values
    s1 = camarilla_s1.values
    
    # Volume confirmation: current volume > 2.0 * 1d volume MA
    volume_confirm = volume > (vol_ma_1d_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    entry_price = 0.0
    
    # Warmup: max of EMA34 (34), Camarilla needs prev day data (effectively ~24h+)
    start_idx = 34  # EMA34 warmup
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        close_val = close[i]
        trend_val = ema34_4h_aligned[i]
        vol_conf = volume_confirm[i]
        r1_val = r1[i]
        s1_val = s1[i]
        
        # Skip if any data not ready or outside session
        if (np.isnan(trend_val) or np.isnan(vol_conf) or np.isnan(r1_val) or np.isnan(s1_val) or not in_session):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 4h EMA34 = uptrend, price < 4h EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Camarilla breakout conditions
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        # Entry conditions: Camarilla breakout in direction of 4h trend + volume + session
        long_entry = long_breakout and is_uptrend and vol_conf
        short_entry = short_breakout and is_downtrend and vol_conf
        
        # Exit conditions: opposite Camarilla touch (mean reversion at H4/L4) or trend reversal
        long_exit = False
        short_exit = False
        if position == 1:
            # Long exit: price touches S1 (mean reversion) or trend turns down
            long_exit = close_val < s1_val or not is_uptrend
        elif position == -1:
            # Short exit: price touches R1 (mean reversion) or trend turns up
            short_exit = close_val > r1_val or not is_downtrend
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0