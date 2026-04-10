#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation
# - Long when 1h close breaks above H3 pivot AND 4h EMA21 > EMA50 (uptrend) AND 1d volume > 1.5x 20-day average
# - Short when 1h close breaks below L3 pivot AND 4h EMA21 < EMA50 (downtrend) AND 1d volume > 1.5x 20-day average
# - Exit when price retouches the pivot point (H3 for longs, L3 for shorts)
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots work well in ranging markets with clear breakouts
# - 4h EMA filter ensures we trade with the higher timeframe trend
# - 1d volume confirmation ensures breakouts have institutional participation

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1h Camarilla pivots (based on previous bar's range)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivots using previous bar's high/low/close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else high[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else low[0]
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else close[0]
    
    # Camarilla pivot levels
    range_val = prev_high - prev_low
    h3 = prev_close + range_val * 1.1 / 4
    l3 = prev_close - range_val * 1.1 / 4
    h4 = prev_close + range_val * 1.1 / 2
    l4 = prev_close - range_val * 1.1 / 2
    
    # Pre-compute 4h EMA trend filter
    df_4h_close = df_4h['close'].values
    ema_21 = pd.Series(df_4h_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    uptrend_4h = ema_21_aligned > ema_50_aligned
    downtrend_4h = ema_21_aligned < ema_50_aligned
    
    # Pre-compute 1d volume confirmation
    df_1d_volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(df_1d_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_spike_1d = df_1d_volume > (1.5 * vol_ma_20)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: 1h close breaks above H3 AND 4h uptrend AND 1d volume spike
            if (close[i] > h3[i] and 
                uptrend_4h[i] and 
                volume_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: 1h close breaks below L3 AND 4h downtrend AND 1d volume spike
            elif (close[i] < l3[i] and 
                  downtrend_4h[i] and 
                  volume_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price retouches the pivot level (H3 for longs, L3 for shorts)
            exit_long = (position == 1 and close[i] <= h3[i])
            exit_short = (position == -1 and close[i] >= l3[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals