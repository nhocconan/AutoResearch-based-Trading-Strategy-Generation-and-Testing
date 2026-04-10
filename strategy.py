#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla breakout with 12h EMA50 trend filter and volume confirmation
# - Uses Camarilla H3/L3 from previous 4h candle as breakout levels
# - 12h EMA50 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation: current volume > 1.5x 20-period average
# - Exit: touch of opposite Camarilla level (L3/H3) or extreme (H4/L4) for reversal
# - Position size: 0.25 to manage risk and minimize fee drag
# - Target: 20-50 trades/year on 4h (80-200 total over 4 years) within proven working range
# - Works in bull/bear: EMA50 on 12h adapts to regime changes, volume filters noise

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute 4h Camarilla levels (based on previous 4h candle's OHLC)
    high_4h = prices['high'].shift(1).values  # previous 4h bar
    low_4h = prices['low'].shift(1).values
    close_4h = prices['close'].shift(1).values
    
    # Camarilla calculation: based on previous bar's range
    rang = high_4h - low_4h
    camarilla_h4 = close_4h + 1.5 * rang * 1.1 / 2
    camarilla_l4 = close_4h - 1.5 * rang * 1.1 / 2
    camarilla_h3 = close_4h + 1.25 * rang * 1.1 / 2
    camarilla_l3 = close_4h - 1.25 * rang * 1.1 / 2
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup for EMA50 and shift
        # Skip if any required data is invalid
        if (np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # 12h trend filter: price > EMA50 = bullish, price < EMA50 = bearish
        bullish_trend = not np.isnan(trend_aligned[i]) and \
                        prices['close'].iloc[i] > trend_aligned[i]
        bearish_trend = not np.isnan(trend_aligned[i]) and \
                        prices['close'].iloc[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > H3 AND bullish trend AND volume confirmation
            if prices['close'].iloc[i] > camarilla_h3[i] and bullish_trend and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: price < L3 AND bearish trend AND volume confirmation
            elif prices['close'].iloc[i] < camarilla_l3[i] and bearish_trend and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or reversal
            # Exit conditions: price touches opposite Camarilla level
            exit_long = prices['close'].iloc[i] < camarilla_l3[i]   # Price breaks below L3 (exit long)
            exit_short = prices['close'].iloc[i] > camarilla_h3[i]  # Price breaks above H3 (exit short)
            
            # Reversal conditions: price hits extreme levels (H4/L4) - counter-trend exit
            reverse_long = prices['close'].iloc[i] >= camarilla_h4[i]  # Price hits H4 (reverse long)
            reverse_short = prices['close'].iloc[i] <= camarilla_l4[i]  # Price hits L4 (reverse short)
            
            exit_condition = (position == 1 and (exit_long or reverse_long)) or \
                           (position == -1 and (exit_short or reverse_short))
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals