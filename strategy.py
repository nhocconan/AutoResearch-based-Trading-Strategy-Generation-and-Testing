#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back through Donchian(20) midline (10-period average)
# Uses volume to confirm breakout strength, EMA50 for trend filter, Donchian for clear entry/exit
# Target: 20-35 trades/year by requiring volume spike + trend alignment + clear breakout

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 12h indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high and low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2  # midline for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        ema50_val = ema50_aligned[i]
        vol_ma = vol_ma_12h_aligned[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        # Find corresponding 12h bar index
        idx_12h = i // 2  # 2 four-hour bars per 12-hour bar
        if idx_12h < len(df_12h):
            volume_12h = df_12h['volume'].iloc[idx_12h]
            volume_confirm = volume_12h > 1.5 * vol_ma
        else:
            volume_confirm = False
        
        if position == 0:
            # Long: break above Donchian high, price > EMA50, volume confirmation
            if price > donch_high_val and price > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, price < EMA50, volume confirmation
            elif price < donch_low_val and price < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below Donchian midline
                if price < donch_mid_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above Donchian midline
                if price > donch_mid_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0