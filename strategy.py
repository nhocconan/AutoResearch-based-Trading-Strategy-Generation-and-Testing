#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Williams Alligator (Jaws, Teeth, Lips) to identify trend direction and strength.
# In trending markets: long when Lips > Teeth > Jaws, short when Lips < Teeth < Jaws.
# Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation: current volume > 1.5x 20-period average.
# Target: 25-40 trades/year by requiring Alligator alignment + 1d trend + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (13, 8, 5 periods)
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(series, period):
        sma = series.rolling(window=period, min_periods=period).mean()
        smma_values = np.full(len(series), np.nan)
        if len(series) >= period:
            smma_values[period-1] = sma.iloc[period-1]
            for i in range(period, len(series)):
                if not np.isnan(sma.iloc[i]):
                    smma_values[i] = (smma_values[i-1] * (period-1) + sma.iloc[i]) / period
                else:
                    smma_values[i] = smma_values[i-1]
        return smma_values
    
    jaws = smma(prices['close'], 13)
    teeth = smma(prices['close'], 8)
    lips = smma(prices['close'], 5)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Get 1d EMA50 for trend filter (updated only after 1d bar closes)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Williams Alligator signals
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaws = teeth[i] > jaws[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaws = teeth[i] < jaws[i]
        
        # 1d trend filter
        price_above_1d_ema = price > ema_50_1d_aligned[i]
        price_below_1d_ema = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaws (bullish alignment) + above 1d EMA + volume
            if lips_above_teeth and teeth_above_jaws and price_above_1d_ema and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaws (bearish alignment) + below 1d EMA + volume
            elif lips_below_teeth and teeth_below_jaws and price_below_1d_ema and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Alligator alignment breaks (Lips < Teeth or Teeth < Jaws)
                if not (lips_above_teeth and teeth_above_jaws):
                    exit_signal = True
                # Also exit if price crosses below 1d EMA (trend change)
                elif price_below_1d_ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Alligator alignment breaks (Lips > Teeth or Teeth > Jaws)
                if not (lips_below_teeth and teeth_below_jaws):
                    exit_signal = True
                # Also exit if price crosses above 1d EMA (trend change)
                elif price_above_1d_ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0