#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1w EMA Trend Filter + Volume Spike
# - Primary: 6h timeframe for balanced trade frequency and fee control
# - HTF: 1w EMA(34) for major trend direction (avoid counter-trend trades)
# - Entry: Williams Alligator signals (Jaw/Teeth/Lips alignment) + 1w EMA trend alignment + volume > 1.5x 20-period MA
# - Exit: Alligator lines reverse (Lips cross Jaw opposite direction) or volume drops below average
# - Position sizing: 0.25 (discrete level)
# - Target: 60-120 total trades over 4 years (15-30/year) - within 6h sweet spot
# - Works in bull/bear: Alligator catches trends, 1w EMA filter avoids major counter-trend moves, volume confirms strength

name = "6h_1w_alligator_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams Alligator on 6h (Smoothed Medians with different periods)
    # Jaw: Blue line - 13-period SMMA shifted 8 bars
    # Teeth: Red line - 8-period SMMA shifted 5 bars  
    # Lips: Green line - 5-period SMMA shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Calculate SMMA for median price (typical price)
    typical_price = (high_6h + low_6h + close_6h) / 3.0
    jaw_raw = smma(typical_price, 13)
    teeth_raw = smma(typical_price, 8)
    lips_raw = smma(typical_price, 5)
    
    # Apply shifts (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set NaN for shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate 1w EMA(34) for major trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h volume moving average (20-period) for volume confirmation
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        # Bullish alignment: Lips > Teeth > Jaw (green above red above blue)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw (green below red below blue)
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_6h[i] > 1.5 * volume_ma_20_6h[i]
        
        # Trend filter: price relative to 1w EMA(34)
        price_above_1w_ema = close_6h[i] > ema_34_1w_aligned[i]
        price_below_1w_ema = close_6h[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish Alligator alignment + price above 1w EMA + volume spike
            if bullish_alignment and price_above_1w_ema and volume_spike:
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish Alligator alignment + price below 1w EMA + volume spike
            elif bearish_alignment and price_below_1w_ema and volume_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines reverse (opposite alignment)
            # 2. Volume drops below average (loss of momentum)
            
            if position == 1:  # Long position
                exit_condition = (
                    bearish_alignment or  # Alligator reversed to bearish
                    volume_6h[i] < volume_ma_20_6h[i]  # Volume dropped below average
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    bullish_alignment or  # Alligator reversed to bullish
                    volume_6h[i] < volume_ma_20_6h[i]  # Volume dropped below average
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals