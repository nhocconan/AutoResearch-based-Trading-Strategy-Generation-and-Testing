#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) + 1w EMA34 trend filter + volume confirmation
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Williams Alligator identifies trending vs ranging markets: 
#   - Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
#   - When Lips > Teeth > Jaw: bullish alignment (all lines rising, price above)
#   - When Lips < Teeth < Jaw: bearish alignment (all lines falling, price below)
#   - When lines intertwine: ranging market (no trade)
# 1w EMA34 determines primary trend bias: long when price > EMA34, short when price < EMA34
# Volume spike (1.5x 20-period average) confirms institutional participation
# Works in bull markets via trend-following entries and bear markets via short signals
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "12h_WilliamsAlligator_1wEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (using smoothed moving average - SMMA)
    # SMMA is similar to EMA but with different smoothing: SMMA_t = (SMMA_{t-1} * (period-1) + price_t) / period
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Jaw: 13-period SMMA, 8 bars shift
    jaw_raw = smma(close, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8 bars
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA, 5 bars shift
    teeth_raw = smma(close, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5 bars
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: 5-period SMMA, 3 bars shift
    lips_raw = smma(close, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3 bars
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Calculate 1w EMA34 trend (prior completed 1w bar's EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 34)  # volume MA(20) and 1w EMA34
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: Lips > Teeth > Jaw (all rising) AND price > 1w EMA34 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Bearish Alligator: Lips < Teeth < Jaw (all falling) AND price < 1w EMA34 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator lines cross (Lips < Teeth) OR price < 1w EMA34 (trend change)
            if lips[i] < teeth[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (Lips > Teeth) OR price > 1w EMA34 (trend change)
            if lips[i] > teeth[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals