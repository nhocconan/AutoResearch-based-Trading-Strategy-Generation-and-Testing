#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based stoploss
# - Long when price breaks above Donchian upper band (20-period high) AND 1d volume > 1.3x 20-bar average
# - Short when price breaks below Donchian lower band (20-period low) AND 1d volume > 1.3x 20-bar average
# - Exit when price touches Donchian middle band (20-period average of high+low)/2 OR ATR stoploss hit
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian channels provide clear structure; volume confirms breakout strength
# - ATR stoploss manages risk without look-ahead (using close-based exit)

name = "4h_1d_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) from 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian middle band: average of upper and lower
    donchian_middle = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.3 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(atr[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper band AND 1d volume spike
            if (prices['close'].iloc[i] > donchian_high[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian lower band AND 1d volume spike
            elif (prices['close'].iloc[i] < donchian_low[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit conditions: 1) price touches Donchian middle band, 2) ATR stoploss hit
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit to middle band (mean reversion)
                if prices['close'].iloc[i] <= donchian_middle[i]:
                    exit_signal = True
                # ATR stoploss: exit if price drops below entry - 2.5 * ATR
                elif prices['close'].iloc[i] < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit to middle band (mean reversion)
                if prices['close'].iloc[i] >= donchian_middle[i]:
                    exit_signal = True
                # ATR stoploss: exit if price rises above entry + 2.5 * ATR
                elif prices['close'].iloc[i] > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals