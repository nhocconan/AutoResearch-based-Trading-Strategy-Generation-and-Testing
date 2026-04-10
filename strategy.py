#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w EMA200 trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1w EMA200 rising AND volume > 1.8x 20-bar avg
# - Short when price breaks below Camarilla L3 level AND 1w EMA200 falling AND volume > 1.8x 20-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses 1w EMA200 for trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-80 trades/year on 1d timeframe (80-320 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance; trend filter improves win rate

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(200) for trend filter
    close_1w_arr = df_1w['close'].values
    ema200_1w = pd.Series(close_1w_arr).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Need at least 2 days of data to compute Camarilla pivots (yesterday's OHLC)
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC for Camarilla calculation
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        prev_close = prices['close'].iloc[i-1]
        
        # Skip if any required data is invalid
        if (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Camarilla pivot levels for today based on yesterday's OHLC
        # Camarilla formulas:
        # Pivot = (prev_high + prev_low + prev_close) / 3
        # H3 = pivot + (prev_high - prev_low) * 1.1 / 4
        # L3 = pivot - (prev_high - prev_low) * 1.1 / 4
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        H3 = pivot + (range_hl * 1.1 / 4.0)
        L3 = pivot - (range_hl * 1.1 / 4.0)
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla H3 AND 1w uptrend with volume spike
            if (prices['close'].iloc[i] > H3 and 
                prices['close'].iloc[i] > ema200_1w_aligned[i] and  # price above 1w EMA200
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Camarilla L3 AND 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < L3 and 
                  prices['close'].iloc[i] < ema200_1w_aligned[i] and  # price below 1w EMA200
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla pivot point
            # Exit when price returns to Camarilla pivot point
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= pivot:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= pivot:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals