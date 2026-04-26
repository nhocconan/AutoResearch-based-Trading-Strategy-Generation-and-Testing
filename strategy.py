#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: On 4h timeframe, Camarilla R1/S1 levels from prior day act as intraday support/resistance. 
Breakout above R1 with volume spike and 1d EMA50 uptrend = long. Breakdown below S1 with volume spike 
and 1d EMA50 downtrend = short. Uses ATR-based trailing stop to limit drawdown in bear markets. 
Designed for 20-50 trades/year with discrete sizing (±0.25) to minimize fee drag and work in 
both bull/bear markets with BTC/ETH edge.
"""

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
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, 
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, 
    # S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use prior day's HLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's data (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), but alignment will handle timing
    
    # Calculate Camarilla R1 and S1 from prior day
    rang = high_1d_prev - low_1d_prev
    r1 = close_1d_prev + rang * 1.1 / 12
    s1 = close_1d_prev - rang * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR for stoploss and position sizing volatility adjustment
    # True Range
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA50 (50), volume MA (20), ATR (14), Camarilla (need prior day)
    start_idx = max(50, 20, 14) + 4  # +4 to ensure 1d bar completion (4h -> 1d: 6 bars per day? Actually 24/4=6, but we use alignment)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        ema_50_val = ema_50_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_val = atr[i]
        
        # Trend condition: 1d EMA50 slope (using current vs 3 periods ago)
        if i >= 3:
            ema_50_prev = ema_50_1d_aligned[i-3]
            ema_50_slope = ema_50_val - ema_50_prev
            uptrend = ema_50_slope > 0
            downtrend = ema_50_slope < 0
        else:
            uptrend = ema_50_val > close_1d[0] if len(close_1d) > 0 else False
            downtrend = ema_50_val < close_1d[0] if len(close_1d) > 0 else False
        
        # Entry conditions
        long_entry = (close_val > r1_val) and vol_spike and uptrend
        short_entry = (close_val < s1_val) and vol_spike and downtrend
        
        # Stoploss conditions: ATR-based trailing stop
        long_stop = False
        short_stop = False
        
        if position == 1:  # Long position
            # Trailing stop: highest high since entry minus 2*ATR
            # We approximate by checking if current close is below recent high - 2*ATR
            recent_high = np.max(high[max(0, i-20):i+1])  # lookback 20 bars (~3.3 days on 4h)
            long_stop = close_val < (recent_high - 2.0 * atr_val)
        elif position == -1:  # Short position
            # Trailing stop: lowest low since entry plus 2*ATR
            recent_low = np.min(low[max(0, i-20):i+1])
            short_stop = close_val > (recent_low + 2.0 * atr_val)
        
        # Exit conditions
        long_exit = long_stop or (close_val < s1_val)  # Also exit if price breaks below S1
        short_exit = short_stop or (close_val > r1_val)  # Also exit if price breaks above R1
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0