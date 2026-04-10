#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# - Long when price breaks above 20-bar high AND volume > 1.5x 20-bar average volume
# - Short when price breaks below 20-bar low AND volume > 1.5x 20-bar average volume
# - Exit when price touches 10-bar EMA (dynamic stop/re-entry)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian channels provide clear breakout levels; volume confirms institutional participation
# - ATR stoploss adapts to volatility, reducing whipsaws in ranging markets

name = "4h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute indicators before loop
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # ATR for dynamic stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 10-period EMA for exit/re-entry signal
    ema_fast = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or np.isnan(ema_fast[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND volume spike
            if high[i] > donchian_high[i] and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volume spike
            elif low[i] < donchian_low[i] and volume_spike[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit/re-entry based on EMA
            exit_signal = False
            if position == 1:  # Long position
                # Exit long if price closes below 10 EMA
                if close[i] < ema_fast[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit short if price closes above 10 EMA
                if close[i] > ema_fast[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                # Re-entry logic: if price breaks Donchian level again with volume
                if position == 1 and high[i] > donchian_high[i] and volume_spike[i]:
                    signals[i] = 0.25  # Add to long
                elif position == -1 and low[i] < donchian_low[i] and volume_spike[i]:
                    signals[i] = -0.25  # Add to short
                else:
                    if position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
    
    return signals