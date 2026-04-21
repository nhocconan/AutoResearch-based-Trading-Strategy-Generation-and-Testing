#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# In trending markets (12h EMA50 > EMA200): take long on upper band break, short on lower band break
# In ranging markets (EMA50 near EMA200): avoid breakouts to reduce false signals
# Uses volume > 1.5x 20-period average for confirmation to ensure conviction
# Target: 20-50 trades/year by requiring strong trend alignment + breakout + volume spike
# Designed to work in both bull (follow trends) and bear (avoid false breakouts in ranging)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-calculate 12h EMA trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Pre-calculate volume moving average (20-period)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        donchian_high = np.max(high_window)
        donchian_low = np.min(low_window)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: 12h EMA50 > EMA200 for uptrend, < for downtrend
        ema_diff = ema_50_aligned[i] - ema_200_aligned[i]
        is_uptrend = ema_diff > 0
        is_downtrend = ema_diff < 0
        # Neutral zone: avoid breakouts when trend is weak
        
        if position == 0:
            if volume_confirm:
                # Only take breakouts in strong trend
                if price > donchian_high and is_uptrend:
                    signals[i] = 0.25
                    position = 1
                elif price < donchian_low and is_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: opposite breakout or trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                if price < donchian_low or (is_downtrend and ema_diff < -ema_50_aligned[i]*0.001):
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > donchian_high or (is_uptrend and ema_diff > ema_50_aligned[i]*0.001):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_12hEMATrend_Volume"
timeframe = "4h"
leverage = 1.0