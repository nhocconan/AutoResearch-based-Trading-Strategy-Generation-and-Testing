#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d Elder Ray filter
# - Long when 6h Williams %R crosses above -80 (oversold bounce) AND 1d Bear Power < 0 (bearish momentum weakening) AND 6h close > 6h EMA(50)
# - Short when 6h Williams %R crosses below -20 (overbought rejection) AND 1d Bull Power > 0 (bullish momentum weakening) AND 6h close < 6h EMA(50)
# - Exit: Williams %R crosses -50 (mean reversion midpoint) or ATR trailing stop (2.5*ATR)
# - Uses 6h for entry timing (Williams %R), 1d for Elder Ray (Bull/Bear Power) to filter counter-trend moves
# - Williams %R captures short-term reversals; Elder Ray confirms underlying momentum shift on higher timeframe
# - EMA(50) filter ensures trades align with intermediate trend to avoid chop
# - Tight entry conditions target 12-30 trades/year to minimize fee drag while maintaining edge
# - Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)

name = "6h_1d_williams_elderray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA(13) for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute 6h EMA(50) for trend filter
    ema50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute ATR for dynamic stoploss (6h)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_6h[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals: cross above -80 (long), cross below -20 (short)
        wr_long_signal = (williams_r[i-1] <= -80) and (williams_r[i] > -80)
        wr_short_signal = (williams_r[i-1] >= -20) and (williams_r[i] < -20)
        wr_exit_signal = (williams_r[i-1] < -50 and williams_r[i] >= -50) or \
                         (williams_r[i-1] > -50 and williams_r[i] <= -50)
        
        # Elder Ray filters: Bear Power < 0 for long, Bull Power > 0 for short
        # (indicates weakening bearish/bullish momentum on 1d)
        elder_long_filter = bear_power_1d_aligned[i] < 0
        elder_short_filter = bull_power_1d_aligned[i] > 0
        
        # EMA(50) trend filter
        uptrend = close[i] > ema50_6h[i]
        downtrend = close[i] < ema50_6h[i]
        
        # Only trade when Elder Ray and EMA filters align
        if elder_long_filter and uptrend:
            if wr_long_signal:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            elif wr_exit_signal and position == 1:
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            elif position == 1:
                # ATR trailing stop: exit if price drops 2.5*ATR from highest high since entry
                if 'highest_high_since_entry' not in locals():
                    highest_high_since_entry = high[i]
                else:
                    highest_high_since_entry = max(highest_high_since_entry, high[i])
                if close[i] < (highest_high_since_entry - 2.5 * atr_6h[i]):
                    if position != 0:  # Only signal on exit
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.0  # Maintain flat
                else:
                    signals[i] = 0.25  # Maintain position
            else:
                signals[i] = 0.0
        elif elder_short_filter and downtrend:
            if wr_short_signal:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            elif wr_exit_signal and position == -1:
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            elif position == -1:
                # ATR trailing stop: exit if price rises 2.5*ATR from lowest low since entry
                if 'lowest_low_since_entry' not in locals():
                    lowest_low_since_entry = low[i]
                else:
                    lowest_low_since_entry = min(lowest_low_since_entry, low[i])
                if close[i] > (lowest_low_since_entry + 2.5 * atr_6h[i]):
                    if position != 0:  # Only signal on exit
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.0  # Maintain flat
                else:
                    signals[i] = -0.25  # Maintain position
            else:
                signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                # Reset tracking variables
                if 'highest_high_since_entry' in locals():
                    del highest_high_since_entry
                if 'lowest_low_since_entry' in locals():
                    del lowest_low_since_entry
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals