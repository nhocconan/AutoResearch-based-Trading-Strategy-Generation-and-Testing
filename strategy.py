#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Williams Alligator: Jaw (13-period SMMA shifted 8 bars), Teeth (8-period SMMA shifted 5 bars), Lips (5-period SMMA shifted 3 bars).
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1d close > EMA50 AND volume > 1.5x 20-bar average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1d close < EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Williams Alligator identifies trend presence and direction, EMA50 filters higher timeframe trend, volume confirms momentum.
# Primary timeframe: 12h, HTF: 1d for EMA trend.

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (SMMA = smoothed moving average)
    def smma(source, period):
        # SMMA is similar to EMA but with alpha = 1/period
        sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(source, np.nan, dtype=np.float64)
        smma_vals[period-1] = sma[period-1]  # First value is SMA
        for i in range(period, len(source)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + source[i]) / period
        return smma_vals
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # Shift right by 8 (future leak fix: we'll align properly)
    jaw[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # Shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # Shift right by 3
    lips[:3] = np.nan
    
    # Align Alligator components to 12h timeframe (already aligned as we used close)
    # But we need to ensure no look-ahead: the shifts are part of the indicator definition
    # No additional alignment needed as we're using same timeframe
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current 12h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for SMMA and EMA
    
    for i in range(start_idx, n):
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Williams Alligator alignment
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish alignment AND price > Lips AND bullish trend AND volume confirmation
            if (bullish_alignment and 
                curr_close > lips[i] and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price < Lips AND bearish trend AND volume confirmation
            elif (bearish_alignment and 
                  curr_close < lips[i] and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish alignment OR price < Lips OR trend turns bearish
            if (bearish_alignment or 
                curr_close < lips[i] or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish alignment OR price > Lips OR trend turns bullish
            if (bullish_alignment or 
                curr_close > lips[i] or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals