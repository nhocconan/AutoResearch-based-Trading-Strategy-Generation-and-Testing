#!/usr/bin/env python3
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
    
    # === 1d data for pivot and ATR ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot and R1/S1 levels (classic formula)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + range_hl * 0.382
    s1 = pivot - range_hl * 0.382
    
    # === True Range and ATR (14-period) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h EMA for trend filter (34-period) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align HTF data to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    atr_14_1h = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_34_1h = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Session filter: 08-20 UTC ===
    # Convert open_time to datetime if needed
    if not isinstance(prices.index, pd.DatetimeIndex):
        times = pd.to_datetime(prices['open_time'])
    else:
        times = prices.index
    hours = times.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(atr_14_1h[i]) or np.isnan(ema_34_1h[i]) or
            np.isnan(volume_spike[i]) or not in_session[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_level = r1_1h[i]
        s1_level = s1_1h[i]
        atr_val = atr_14_1h[i]
        ema_val = ema_34_1h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price drops below S1
            if price < s1_level:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1
            if price > r1_level:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike, above EMA34
            if price > r1_level and vol_spike and price > ema_val:
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike, below EMA34
            elif price < s1_level and vol_spike and price < ema_val:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Pivot_R1_S1_Breakout_Volume_EMA34"
timeframe = "1h"
leverage = 1.0