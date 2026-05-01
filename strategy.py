#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume spike confirmation.
# Long when: price breaks above R1 AND 4h close > 4h EMA50 AND 1h volume > 1.5x 20-period average
# Short when: price breaks below S1 AND 4h close < 4h EMA50 AND 1h volume > 1.5x 20-period average
# Uses discrete sizing 0.20. Target: 15-37 trades/year on 1h.
# Camarilla pivots provide intraday support/resistance, 4h EMA50 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) by trading with the aligned 4h trend.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1h data ONCE before loop for Camarilla calculation (need full session OHLC)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for 1h using previous bar's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1h['close'].shift(1).values
    prev_high = df_1h['high'].shift(1).values
    prev_low = df_1h['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h primary timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h volume average (20-period) for volume spike confirmation
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_1h[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume spike: current 1h volume > 1.5x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 1.5)
        
        # 4h trend filter: price above/below EMA50
        uptrend_4h = curr_close > curr_ema_50
        downtrend_4h = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R1 AND 4h uptrend AND volume spike
            if (curr_close > curr_r1 and 
                uptrend_4h and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND 4h downtrend AND volume spike
            elif (curr_close < curr_s1 and 
                  downtrend_4h and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below S1 (reversal) OR 4h trend turns down
            if (curr_close < curr_s1 or 
                not uptrend_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 (reversal) OR 4h trend turns up
            if (curr_close > curr_r1 or 
                not downtrend_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals