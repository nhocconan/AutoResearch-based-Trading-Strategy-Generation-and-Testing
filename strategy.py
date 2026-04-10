#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above H3 level AND 4h EMA50 rising AND volume > 2.0x 20-bar avg
# - Short when price breaks below L3 level AND 4h EMA50 falling AND volume > 2.0x 20-bar avg
# - Exit when price returns to Pivot level (mean reversion to equilibrium)
# - Uses 4h EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.20) to minimize fee churn
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots work well in ranging markets; trend filter adds directional bias in trends

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous bar's range
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # H3, L3 levels (most important for breakouts)
    h3 = close_4h + (range_4h * 1.1 / 4)
    l3 = close_4h - (range_4h * 1.1 / 4)
    # Pivot level for exit
    piv = pivot
    
    # Align HTF levels to LTF
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    piv_aligned = align_htf_to_ltf(prices, df_4h, piv)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h_arr = df_4h['close'].values
    ema50_4h = pd.Series(close_4h_arr).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(piv_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if not in_session.iloc[i]:
            # Outside session: hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h uptrend with volume spike
            if (prices['close'].iloc[i] > h3_aligned[i] and 
                prices['close'].iloc[i] > ema50_4h_aligned[i] and  # price above 4h EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h downtrend with volume spike
            elif (prices['close'].iloc[i] < l3_aligned[i] and 
                  prices['close'].iloc[i] < ema50_4h_aligned[i] and  # price below 4h EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= piv_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= piv_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals