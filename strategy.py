#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# Donchian channels provide clear trend-following structure with defined breakout levels
# Volume confirmation (>1.4x average) filters weak breakouts to reduce false signals
# ATR stoploss (2.5x ATR) manages risk and allows trends to run
# Works in bull/bear: captures strong moves in trending markets, volume confirms legitimacy
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Donchian20_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period) from previous bar
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Need previous bar's levels to avoid look-ahead
    donchian_upper_prev = np.roll(donchian_upper, 1)
    donchian_lower_prev = np.roll(donchian_lower, 1)
    donchian_upper_prev[0] = np.nan
    donchian_lower_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > donchian_upper_prev
    breakout_down = close < donchian_lower_prev
    
    # Volume confirmation: volume > 1.4x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.4 * vol_ma_20)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(100, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper_prev[i]) or 
            np.isnan(donchian_lower_prev[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price above Donchian upper
                if curr_breakout_up:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
                # Bearish breakout: price below Donchian lower
                elif curr_breakout_down:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
        
        elif position == 1:  # Long position
            # Stoploss: price closes below entry - 2.5 * ATR_at_entry
            # Take profit: price closes above Donchian upper (trailing)
            stop_loss = entry_price - 2.5 * atr_at_entry
            if curr_close < stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.5 * ATR_at_entry
            # Take profit: price closes below Donchian lower (trailing)
            stop_loss = entry_price + 2.5 * atr_at_entry
            if curr_close > stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals