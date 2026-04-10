#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray combination with volume confirmation
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when: Alligator aligned bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND volume > 1.5x volume SMA(20)
# - Short when: Alligator aligned bearish (Lips < Teeth < Jaw) AND Bear Power < 0 AND volume > 1.5x volume SMA(20)
# - Exit: Alligator convergence (|Lips - Jaw| < 0.1 * ATR) OR ATR trailing stop (2.0x ATR)
# - Uses 1d EMA13 for Elder Ray calculation (HTF for stability)
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Williams Alligator identifies trend, Elder Ray measures power, volume confirms conviction
# - Works in both bull and bear markets by capturing strong directional moves with confirmation

name = "4h_1d_alligator_elder_ray_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d EMA13 for Elder Ray (HTF indicator)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Williams Alligator components on 4h data
    # Jaw: 13-period SMMA, smoothed by 8 periods
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMMA, smoothed by 5 periods
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMMA, smoothed by 3 periods
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean().values
    
    # Calculate Elder Ray components
    # Bull Power = High - EMA13
    bull_power = high - ema13_1d_aligned
    # Bear Power = Low - EMA13
    bear_power = low - ema13_1d_aligned
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume SMA(20) for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Williams Alligator alignment conditions
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        if position == 0:  # Flat - look for entry
            # Long: Bullish Alligator alignment AND Bull Power > 0 AND volume confirmation
            if bullish_alignment and bull_power_positive and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: Bearish Alligator alignment AND Bear Power < 0 AND volume confirmation
            elif bearish_alignment and bear_power_negative and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # ATR trailing stop: exit if price drops 2.0*ATR below highest high since entry
            # For simplicity, we'll use a close-based trailing stop
            # Track highest close since entry using a simple approach
            if i == 20:  # Initialize on first valid bar
                highest_close_since_entry = close[i]
            else:
                highest_close_since_entry = max(getattr(generate_signals, 'highest_close_since_entry', close[i-1]), close[i])
            
            trailing_stop = highest_close_since_entry - 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR Alligator convergence (|Lips - Jaw| < 0.1 * ATR)
            alligator_convergence = np.abs(lips[i] - jaw[i]) < 0.1 * atr[i]
            exit_condition = (close[i] < trailing_stop) or alligator_convergence
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset for next entry
                if hasattr(generate_signals, 'highest_close_since_entry'):
                    delattr(generate_signals, 'highest_close_since_entry')
            else:
                signals[i] = 0.25
                # Update highest close since entry
                highest_close_since_entry = max(getattr(generate_signals, 'highest_close_since_entry', close[i-1]), close[i])
                setattr(generate_signals, 'highest_close_since_entry', highest_close_since_entry)
        else:  # position == -1 (Short position) - look for exit
            # ATR trailing stop: exit if price rises 2.0*ATR above lowest low since entry
            # Track lowest close since entry using a simple approach
            if i == 20:  # Initialize on first valid bar
                lowest_close_since_entry = close[i]
            else:
                lowest_close_since_entry = min(getattr(generate_signals, 'lowest_close_since_entry', close[i-1]), close[i])
            
            trailing_stop = lowest_close_since_entry + 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR Alligator convergence (|Lips - Jaw| < 0.1 * ATR)
            alligator_convergence = np.abs(lips[i] - jaw[i]) < 0.1 * atr[i]
            exit_condition = (close[i] > trailing_stop) or alligator_convergence
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset for next entry
                if hasattr(generate_signals, 'lowest_close_since_entry'):
                    delattr(generate_signals, 'lowest_close_since_entry')
            else:
                signals[i] = -0.25
                # Update lowest close since entry
                lowest_close_since_entry = min(getattr(generate_signals, 'lowest_close_since_entry', close[i-1]), close[i])
                setattr(generate_signals, 'lowest_close_since_entry', lowest_close_since_entry)
    
    return signals