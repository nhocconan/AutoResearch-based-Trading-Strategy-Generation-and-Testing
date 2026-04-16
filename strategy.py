#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation and 1w trend filter.
# Long when price breaks above R4 AND 1w EMA50 uptrend AND volume > 2x 20-period average.
# Short when price breaks below S4 AND 1w EMA50 downtrend AND volume > 2x 20-period average.
# Exit when price returns to the 1d pivot point (PP).
# Uses discrete position size 0.25. 1d/1w filters provide signal direction, 6h provides entry timing.
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (using previous day) ===
    # Calculate PP, R4, S4 from previous 1d bar
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s4_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (6h)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        pp = pp_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        ema50 = ema50_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 6h volume average aligned
        vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price <= pp:  # Exit when price returns to or below pivot point
                exit_signal = True
        
        elif position == -1:  # Short position
            if price >= pp:  # Exit when price returns to or above pivot point
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine 1w trend: bullish if close > EMA50, bearish if close < EMA50
            # Need previous bar's 1w close and EMA for trend determination
            if i >= warmup + 1:
                prev_ema50 = ema50_aligned[i-1]
                # Get previous 1w close aligned (need to access df_1w close properly)
                # Simpler: use current and previous aligned EMA to infer trend
                if close[i-1] > prev_ema50:  # Previous bar close above EMA50 = uptrend
                    trend_bullish = True
                    trend_bearish = False
                elif close[i-1] < prev_ema50:  # Previous bar close below EMA50 = downtrend
                    trend_bullish = False
                    trend_bearish = True
                else:
                    trend_bullish = False
                    trend_bearish = False
            else:
                trend_bullish = False
                trend_bearish = False
            
            # LONG: Price breaks above R4 AND 1w uptrend AND volume > 2x 20-period avg
            if (price > r4) and trend_bullish and (vol > 2.0 * vol_ma_20_6h[i]):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S4 AND 1w downtrend AND volume > 2x 20-period avg
            elif (price < s4) and trend_bearish and (vol > 2.0 * vol_ma_20_6h[i]):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dCamarillaR4S4_1wEMA50_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0