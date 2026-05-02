#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla R4/S4 levels represent extreme support/resistance where breaks indicate strong institutional momentum
# 1w EMA50 provides long-term trend filter to avoid counter-trend whipsaws in ranging markets
# Volume spike (2.5x 20-period average) confirms breakout conviction with high conviction
# Targets 30-100 trades over 4 years (7-25/year) for 1d timeframe
# Works in bull markets by capturing continuation breaks and in bear markets by capturing panic-driven breakdowns
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining adequate exposure

name = "1d_Camarilla_R4_S4_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #          S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use the previous completed 1d bar to avoid look-ahead
    high_1d = high
    low_1d = low
    close_1d = close
    
    camarilla_high = close_1d + 1.5 * (high_1d - low_1d)  # R4 level
    camarilla_low = close_1d - 1.5 * (high_1d - low_1d)   # S4 level
    
    camarilla_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_1d, 'low': low_1d, 'close': close_1d}), camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_1d, 'low': low_1d, 'close': close_1d}), camarilla_low)
    
    # Calculate volume spike (2.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R4 + price > 1w EMA50 + volume spike
            if close[i] > camarilla_high_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S4 + price < 1w EMA50 + volume spike
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S4 (reversal signal)
            if close[i] < camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R4 (reversal signal)
            if close[i] > camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals