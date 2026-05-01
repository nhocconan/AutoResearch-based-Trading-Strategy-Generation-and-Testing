#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation.
# Long when Bear Power < 0 (bulls in control) AND 1d close > EMA34 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when Bull Power > 0 (bears in control) AND 1d close < EMA34 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Elder Ray measures bull/bear power relative to EMA13, EMA34 filters higher-timeframe trend, volume spike confirms conviction.
# Primary timeframe: 6h, HTF: 1d for EMA trend filter.

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA34 trend filter
    prev_close_1d = df_1d['close'].values
    ema_34 = pd.Series(prev_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and indicators
    
    for i in range(start_idx, n):
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Elder Ray signals
        bulls_in_control = curr_bear_power < 0  # Bear Power negative = bulls in control
        bears_in_control = curr_bull_power > 0  # Bull Power positive = bears in control
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bears NOT in control (bear power < 0) AND bullish trend AND volume confirmation
            if (bulls_in_control and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bulls NOT in control (bull power > 0) AND bearish trend AND volume confirmation
            elif (bears_in_control and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bears take control (bear power > 0) OR trend turns bearish
            if (curr_bear_power > 0 or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bulls take control (bull power < 0) OR trend turns bullish
            if (curr_bull_power < 0 or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals